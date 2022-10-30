from mir.io.feature_io_base import *
from mir.common import PACKAGE_PATH
import numpy as np
from numpy.linalg import norm
from osu_common import SPEED_TYPES,SPEED_BOUND

class OsuTagIO(FeatureIO):
    def read(self, filename, entry):
        return pickle_read(self,filename)

    def write(self, data, filename, entry):
        return pickle_write(self,data,filename)

    def visualize(self, data, filename, entry, override_sr):
        labels = ['onset','is_normal','is_slider_start','is_slider_end','is_slider_reverse','is_spinner','position_x','position_y',
                  'speed','angle_x','angle_y','delta_angle_x','delta_angle_y','center_dist','inner_beat_pos','tp_index','beat_index']+SPEED_TYPES
        f = open(os.path.join(PACKAGE_PATH,'data/spectrogram_template.svl'), 'r')
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        n_frame=int(data[-1][0]/1000.*sr/win_shift)+20
        onsets=[int(hit[0]/1000.*sr/win_shift) for hit in data]
        abs_pos=np.array([hit[2:4] for hit in data])
        result=np.zeros((n_frame,len(labels)))
        old_angle=np.array([1.0,0.0])
        for i,hit in enumerate(data):
            is_last=i+1==len(data)
            start=onsets[i]
            end=n_frame if is_last else onsets[i+1]
            result[start,0]=1
            result[start:end,hit[1]]=1
            result[start:end,6]=hit[2]/512 if hit[2]>=0 else -0.01
            result[start:end,7]=hit[3]/384 if hit[3]>=0 else -0.01
            center_dist=norm(abs_pos[i]-np.array([256,192]))/norm(np.array([256,192])) if hit[2]>=0 else -0.01
            result[start:end,13]=center_dist
            result[start:end,14]=hit[6]/hit[7]
            result[start:end,15]=hit[4]
            result[start:end,16]=hit[5]
            # calculate angle and speed
            #speed=-0.01
            angle=np.array([-0.01,-0.01])
            delta_angle=np.array([-0.01,-0.01])
            if(not is_last):
                next_hit=data[i+1]
                delta=abs_pos[i+1]-abs_pos[i]
                #speed=norm(delta)/(next_hit[0]-hit[0]) # pixel/ms
                if(norm(delta)<=1e-6):
                    angle=old_angle
                else:
                    angle=delta/norm(delta)
                delta_angle=np.arctan2(angle[0],angle[1])-np.arctan2(old_angle[0],old_angle[1])
                delta_angle=np.array([np.cos(delta_angle),np.sin(delta_angle)])
                old_angle=angle
            speed=hit[8]
            result[start:end,8]=speed
            result[start:end,9:11]=angle
            result[start:end,11:13]=delta_angle
            if(speed>=-1e-6):
                result[start:end,np.searchsorted(SPEED_BOUND,speed)+17]=1.0
        max_speed=result[:,8].max()
        result[:,8]/=max_speed
        result[:,15]/=(result[:,15].max()+1)
        result[:,16]/=(result[:,16].max()+1)
        labels[8]='%s (%.2fx)'%(labels[8],max_speed)


        content = f.read()
        f.close()
        content = content.replace('[__SR__]', str(sr))
        content = content.replace('[__WIN_SHIFT__]', str(win_shift))
        content = content.replace('[__SHAPE_1__]', str(len(labels)))
        content = content.replace('[__COLOR__]', str(1))
        content = content.replace('[__DATA__]',create_svl_3d_data(labels,result))
        f=open(filename,'w')
        f.write(content)
        f.close()


    def get_visualize_extention_name(self):
        return "svl"