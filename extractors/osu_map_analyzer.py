from mir.extractors import ExtractorBase
from mir import io
import numpy as np
from io_new.downbeat_io import DownbeatIO
from io_new.osu_tag_io import OsuTagIO
from osu_parser.beatmapparser import BeatmapParser
from mir.extractors.misc import FrameCount

#POS_STATISTICS=[]

QUANTIZED_DIST=[0,1,1,1,1,3,1,5,2,3,5,7,1]
QUANTIZED_UNIT=[1,8,6,4,3,8,2,8,3,4,6,8,1]

QUANTIZED_POSITION=[a/b for (a,b) in zip(QUANTIZED_DIST,QUANTIZED_UNIT)]
QUANTIZED_DELIMITER=np.array([(a+b)/2 for (a,b) in zip(QUANTIZED_POSITION[1:],QUANTIZED_POSITION[:-1])])

class OsuHitObjects(ExtractorBase):

    def get_feature_class(self):
        return OsuTagIO

    def extract(self,entry,**kwargs):
        map=entry.dict[kwargs['source']].get(entry)
        parser=BeatmapParser()
        parser.parseFile(entry.dict[kwargs['source']].filepath)
        beatmap=parser.build_beatmap()
        result=[]
        special_mode=False
        for hit in beatmap['hitObjects']:
            hit_pos=hit['startTime']
            hit_type=hit['object_name']
            def norm(x):
                return x if x>=0 else 0

            if(hit_type=='circle'):
                result.append([hit_pos,1,norm(hit['position'][0]),norm(hit['position'][1])])
            elif(hit_type=='slider'):
                slider_start=hit_pos
                segment_count=hit['repeatCount']
                slider_end=hit['end_time']
                pos=[tuple(hit['position']),tuple(hit['end_position'])]
                for i,slider_time in enumerate(np.linspace(slider_start,slider_end,segment_count+1)):
                    result.append([slider_time,2 if i==0 else 3 if i==segment_count else 4,
                                               norm(pos[i%2][0]),norm(pos[i%2][1])])
            elif(hit_type=='spinner'):
                spinner_end=hit['end_time']
                result.append([hit_pos,5,-1,-1])
                result.append([spinner_end,6,-1,-1])
            elif(hit_type=='mania_slider'): # this is only a test
                slider_end=hit['end_time']
                pos=hit['position']
                result.append([hit_pos,2,norm(pos[0]),norm(pos[1])])
                result.append([slider_end,3,norm(pos[0]),norm(pos[1])])
                special_mode=True
            else:
                raise NotImplementedError("no such hit object type: %s"%hit_type)
        if(not special_mode):
            for i in range(len(result)-1):
                assert(result[i+1][0]>=result[i][0])
        else:
            result.sort(key=lambda x:x[0])

        qt_result=[]
        current_beat_length=-1.0
        current_beat_start=0.0
        p=0
        tempo_idx=-1
        timing_points=[tp_str.split(',') for tp_str in map.timingpoints]
        timing_points_pos=[float(t[0]) for t in timing_points]
        prev_position_x=-1
        prev_position_y=-1
        prev_pos=0
        for i,hit in enumerate(result):
            # beat position related
            hit_pos=hit[0]
            object_type=hit[1]
            position_x=hit[2]
            position_y=hit[3]
            while((p<len(timing_points_pos) and timing_points_pos[p]<=hit_pos+1e-6) or p==0):
                tokens=timing_points[p]
                beat_length=float(tokens[1])
                if(beat_length>=0):
                    current_beat_length=beat_length
                    current_beat_start=timing_points_pos[p]
                    tempo_idx+=1
                p+=1
            approx_pos=(hit_pos-current_beat_start)/current_beat_length
            inner_beat_pos=approx_pos%1.0
            beat_id=int(np.round(approx_pos-inner_beat_pos))
            idx=np.searchsorted(QUANTIZED_DELIMITER,inner_beat_pos)
            if(idx==len(QUANTIZED_POSITION)-1):
                idx=0
                beat_id+=1
            # speed
            if(prev_position_x==-1 or position_x==-1):
                speed=-1
            else:
                delta_x=position_x-prev_position_x
                delta_y=position_y-prev_position_y
                delta=np.sqrt(delta_x*delta_x+delta_y*delta_y)
                speed=delta/(hit_pos-prev_pos)
            prev_position_x=position_x
            prev_position_y=position_y
            prev_pos=hit_pos
            qt_result.append([hit_pos,object_type,position_x,position_y,
                               tempo_idx,beat_id,QUANTIZED_DIST[idx],QUANTIZED_UNIT[idx],speed])
        #print(create_subbeat_array(timing_points,result,4))
        return qt_result

def create_subbeat_array(timing_points,hit_objects,beat_division,n_time_ms=-1, safe_interval=0.02):
    timing_points_pos=[float(t[0]) for t in timing_points]
    tempo_idx=-1
    current_beat_length=-1.0
    current_beat_start=0.0
    beat_ids=[]
    valid_segs=[]
    for p,tp in enumerate(timing_points):
        tokens=timing_points[p]
        beat_length=float(tokens[1])
        if(beat_length>=0):
            current_beat_length=beat_length
            current_beat_start=timing_points_pos[p]
            time_signature=int(tokens[2]) if len(tokens)>2 else 4
            valid_segs.append([current_beat_start,current_beat_length,time_signature])
            tempo_idx+=1
    start_seg_id=hit_objects[0][4]
    end_seg_id=hit_objects[-1][4]
    result=[]
    for seg_id in range(start_seg_id,end_seg_id+1):
        current_beat_start=valid_segs[seg_id][0]
        current_beat_length=valid_segs[seg_id][1]
        if(seg_id==start_seg_id):
            start_subbeat_id=min(0,hit_objects[0][5]*beat_division) # be careful with minus beat position
        else:
            start_subbeat_id=0
        if(seg_id==end_seg_id):
            if(n_time_ms>0):
                end_subbeat_id=int(np.floor(
                ((n_time_ms-valid_segs[seg_id][0])/current_beat_length-0.02)*beat_division))
            else:
                end_subbeat_id=hit_objects[-1][5]*beat_division+beat_division # keep whole ending beat
        else:
            end_subbeat_id=int(np.floor(
                ((valid_segs[seg_id+1][0]-valid_segs[seg_id][0])/current_beat_length-safe_interval)*beat_division
                # 0.02083=1/48 is the minimal possible safe interval between any numbers in QUANTIZED_POSITION
            ))
        #print('Sub-beat range for seg %d: %d-%d'%(seg_id,start_subbeat_id,end_subbeat_id))
        result+=[
            [
                (i*(current_beat_length/beat_division)+current_beat_start)/1000.0,
                seg_id,i,2 if (i%(beat_division*valid_segs[seg_id][2])==0) else 1 if (i%beat_division==0) else 0
            ] for i in range(start_subbeat_id,end_subbeat_id+1)
        ]
    return result


class OsuTimingPointsToSubBeat(ExtractorBase):

    def get_feature_class(self):
        return DownbeatIO

    def extract(self,entry,**kwargs):
        beat_division=kwargs['beat_div'] if 'beat_div' in kwargs else 4
        safe_interval=kwargs['safe_interval'] if 'safe_interval' in kwargs else 0.02
        map=entry.dict[kwargs['source']].get(entry)
        hit_objects=entry.apply_extractor(OsuHitObjects,source=kwargs['source'])
        timing_points=[tp_str.split(',') for tp_str in map.timingpoints]
        subbeat_array=create_subbeat_array(timing_points,hit_objects,beat_division, safe_interval=safe_interval)
        return np.array([[token[0],token[3]] for token in subbeat_array])

class OsuTimingPointsToFramedSubBeat(ExtractorBase):
    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        beat_division=kwargs['beat_div'] if 'beat_div' in kwargs else 4
        use_song_ending=kwargs['use_song_ending'] if 'use_song_ending' in kwargs else False
        map=entry.dict[kwargs['source']].get(entry)
        hit_objects=entry.apply_extractor(OsuHitObjects,source=kwargs['source'])
        n_frame=entry.apply_extractor(FrameCount,source='cqt')
        hop_length=entry.prop.hop_length
        sr=entry.prop.sr
        timing_points=[tp_str.split(',') for tp_str in map.timingpoints]
        if(use_song_ending):
            subbeat_array=create_subbeat_array(timing_points,hit_objects,beat_division,n_frame*hop_length/sr*1000)
        else:
            subbeat_array=create_subbeat_array(timing_points,hit_objects,beat_division)
        result=np.zeros((n_frame),dtype=np.int8)
        for token in subbeat_array:
            frame=int(np.round(token[0]*sr/hop_length))
            if(frame>=0 and frame<n_frame):
                result[frame]=token[3]
        return result


class OsuSubBeatLevelHitObject(ExtractorBase):
    def get_feature_class(self):
        return io.RegionalSpectrogramIO

    def extract(self,entry,**kwargs):
        beat_division=kwargs['beat_div'] if 'beat_div' in kwargs else 4
        use_song_ending=kwargs['use_song_ending'] if 'use_song_ending' in kwargs else False

        map=entry.dict[kwargs['source']].get(entry)
        hit_objects=entry.apply_extractor(OsuHitObjects,cache_enabled=False,source=kwargs['source'])
        hop_length=entry.prop.hop_length
        sr=entry.prop.sr
        timing_points=[tp_str.split(',') for tp_str in map.timingpoints]
        if(use_song_ending):
            n_frame=entry.apply_extractor(FrameCount,source='cqt')
            subbeat_array=create_subbeat_array(timing_points,hit_objects,beat_division,n_frame*hop_length/sr*1000)
        else:
            subbeat_array=create_subbeat_array(timing_points,hit_objects,beat_division)
        timing=[t[0] for t in subbeat_array]
        frame_ids=[int(np.round(t[0]*sr/hop_length)) for t in subbeat_array]
        n_subbeat=len(subbeat_array)
        result=np.zeros((n_subbeat,7))
        p=0
        current_state=0 # 0: none 1: slider 2: spinner

        for hit in hit_objects:
            hit_type=hit[1]
            hit_pos_x=hit[2]
            hit_pos_y=hit[3]
            tp_idx=hit[4]
            beat_id=hit[5]
            quant_up=hit[6]
            quant_down=hit[7]
            speed=hit[8]
            if(beat_division%quant_down!=0):
                # unexpected situation, put invalid flags on previous sub-beat
                subbeat_id=beat_id*beat_division+int(np.floor((quant_up*beat_division)/quant_down))
                erase=True
            else:
                subbeat_id=beat_id*beat_division+quant_up*beat_division//quant_down
                erase=False
            #print('Target: %d %d'%(tp_idx,subbeat_id))
            while(tp_idx>subbeat_array[p][1] or (tp_idx==subbeat_array[p][1] and subbeat_id>subbeat_array[p][2])):
                #print('Current: %d %d'%(subbeat_array[p][1],subbeat_array[p][2]))
                p+=1
                result[p][5]=current_state
            #print('Current: %d %d'%(subbeat_array[p][1],subbeat_array[p][2]))
            # fix a minor error where a hitobject happens slightly before a timing point
            if(tp_idx<subbeat_array[p][1]):
                assert(p>0)
                assert(subbeat_array[p-1][2]+1==subbeat_id)
                assert(subbeat_array[p-1][1]==tp_idx)
                tp_idx=subbeat_array[p][1]
                subbeat_id=subbeat_array[p][2]
                #print('Fixed Target: %d %d'%(tp_idx,subbeat_id))
            assert(tp_idx==subbeat_array[p][1])
            assert(subbeat_id==subbeat_array[p][2])
            if(erase):
                result[p][1]=-1
            else:
                result[p][1]=hit_type
                result[p][2]=hit_pos_x
                result[p][3]=hit_pos_y
            result[p][4]=speed
            if(hit_type==2):
                current_state=1
            elif(hit_type==5):
                current_state=2
            elif(hit_type==3 or hit_type==6):
                current_state=0
        result[:,0]=frame_ids
        result[:,6]=[t[3] for t in subbeat_array]
        return timing,result

class OsuManiaSubBeatLevelHitObject(ExtractorBase):
    def get_feature_class(self):
        return io.RegionalSpectrogramIO

    def extract(self,entry,**kwargs):
        num_keys=kwargs['keys']
        beat_division=kwargs['beat_div'] if 'beat_div' in kwargs else 4
        use_song_ending=kwargs['use_song_ending'] if 'use_song_ending' in kwargs else False

        map=entry.dict[kwargs['source']].get(entry)
        hit_objects=entry.apply_extractor(OsuHitObjects,cache_enabled=False,source=kwargs['source'])
        hop_length=entry.prop.hop_length
        sr=entry.prop.sr
        timing_points=[tp_str.split(',') for tp_str in map.timingpoints]
        if(use_song_ending):
            n_frame=entry.apply_extractor(FrameCount,source='cqt')
            subbeat_array=create_subbeat_array(timing_points,hit_objects,beat_division,n_frame*hop_length/sr*1000)
        else:
            subbeat_array=create_subbeat_array(timing_points,hit_objects,beat_division)
        timing=[t[0] for t in subbeat_array]
        frame_ids=[int(np.round(t[0]*sr/hop_length)) for t in subbeat_array]
        n_subbeat=len(subbeat_array)
        result=np.zeros((n_subbeat,num_keys+1))
        for ichannel in range(num_keys):
            p=0
            current_state=0 # 0: none 4: slider 5: spinner
            for hit in hit_objects:
                hit_type=hit[1]
                hit_channel=int(np.floor(hit[2]*num_keys/512))
                if(hit_channel<0):
                    hit_channel=0
                if(hit_channel>=num_keys):
                    hit_channel=num_keys-1
                if(hit_channel!=ichannel):
                    continue
                tp_idx=hit[4]
                beat_id=hit[5]
                quant_up=hit[6]
                quant_down=hit[7]
                # print(hit_type,hit_channel,tp_idx,beat_id,quant_up,quant_down)
                if(beat_division%quant_down!=0):
                    # unexpected situation, put invalid flags on previous sub-beat
                    subbeat_id=beat_id*beat_division+int(np.floor((quant_up*beat_division)/quant_down))
                    erase=True
                else:
                    subbeat_id=beat_id*beat_division+quant_up*beat_division//quant_down
                    erase=False
                #print('Target: %d %d'%(tp_idx,subbeat_id))
                while(tp_idx>subbeat_array[p][1] or (tp_idx==subbeat_array[p][1] and subbeat_id>subbeat_array[p][2])):
                    #print('Current: %d %d'%(subbeat_array[p][1],subbeat_array[p][2]))
                    p+=1
                    result[p][ichannel+1]=current_state
                #print('Current: %d %d'%(subbeat_array[p][1],subbeat_array[p][2]))
                # fix a minor error where a hitobject happens slightly before a timing point
                if(tp_idx<subbeat_array[p][1]):
                    # assert(p>0)
                    # assert(subbeat_array[p-1][2]+1==subbeat_id)
                    # assert(subbeat_array[p-1][1]==tp_idx)
                    tp_idx=subbeat_array[p][1]
                    subbeat_id=subbeat_array[p][2]
                    #print('Fixed Target: %d %d'%(tp_idx,subbeat_id))
                assert(tp_idx==subbeat_array[p][1])
                assert(subbeat_id==subbeat_array[p][2])
                if(erase):
                    result[p][ichannel+1]=-1
                else:
                    result[p][ichannel+1]=hit_type
                if(hit_type==2):
                    current_state=4
                elif(hit_type==5):
                    assert(False)
                elif(hit_type==3 or hit_type==6):
                    current_state=0
            result[:,0]=frame_ids
            # result[:,1]=[t[3] for t in subbeat_array]
        return timing,result

