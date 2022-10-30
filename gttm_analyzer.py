import os
from settings import RWC_DATASET_PATH
import pretty_midi
from data_preprocess import prepare_quantization, get_piano_roll
import numpy as np
import music21
import requests
import re
import xml.etree.ElementTree
from tcn_downbeat_eval import get_rolls, eval_beat_result

def get_split(data_file, split):

    f = open('./data/%s.split.txt.names' % data_file, 'r')
    tokens = [line.strip().split(',') for line in f.readlines() if line.strip() != '']
    f.close()
    if (split == 'train'):
        return tokens[0]
    elif (split == 'val'):
        return tokens[1]
    elif (split == 'test'):
        return tokens[2]
    else:
        raise Exception('No such split')

def pre_quantize(file, subbeat_count=4):
    try:
        print('Processing ' + file)
        midi = pretty_midi.PrettyMIDI(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file))
    except:
        print('Error loading ' + file)
        return
    n_subbeat, downbeat_bins, boundaries, subbeat_time = prepare_quantization(midi, subbeat_count)
    
    result_ins = pretty_midi.Instrument(0)
    for ins in midi.instruments:
        if ('mel' in ins.name.lower() or 'vocal' in ins.name.lower()):
            for note in ins.notes:
                start_bin = np.searchsorted(boundaries, note.start)
                end_bin = np.searchsorted(boundaries, note.end)
                if (end_bin == start_bin):
                    end_bin += 1
                result_ins.notes.append(pretty_midi.Note(pitch=note.pitch, velocity=note.velocity,
                                                         start=start_bin / subbeat_count, end=end_bin / subbeat_count))
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=60)
    new_midi.instruments.append(result_ins)
    new_midi.write('output/gttm/' + file)
    music21_midi = music21.converter.parse('output/gttm/' + file)
    music21_midi.write("musicxml", fp='output/gttm/' + file + '.xml')

def split_params(params, sessionID):
    result = {'sessionID': sessionID}
    params = params.split('&')
    for param in params:
        tokens = param.split('=')
        result[tokens[0]] = tokens[1]
    return result
def reg_test(music_xml_path):
    f = open(music_xml_path + '.gpr.xml', 'r')
    content = ''.join(f.readlines())
    f.close()
    content2 = re.sub(r'^\s+<supplementary[^>]+/>$', '', content)
    f = open(music_xml_path + '.gpr4.xml', 'w')
    f.write(content2)
    f.close()

def get_mpr(music_xml_path):
    f = open(music_xml_path, 'r')
    content = ''.join(f.readlines())
    f.close()
    # part_id = content.split('<score-part id="')[1].split('"')[0]
    r = requests.get('http://gttm.jp/gttm_analysis_scripts/ver1_3/registerSession.php')
    sessionId = r.text
    r = requests.post('http://gttm.jp/gttm_analysis_scripts/ver1_3/submitMXML.php', {'sessionID': sessionId, 'mxmlData': content})
    print('xml submission', r.text)
    params = 'w_sigma=0.05&w_GPR6_length=0.5&w_GPR6_start_end=0.5&w_GPR236=0.5&w_GPR6_rithm_register=0.5&w_GPR2b=0.5&w_GPR7T=0.5&w_GPR2a=0.5&t_GPR6=0.5&w_GPR7P=0.5&t_GPR4=0.5&w_GPR3d=0.5&w_GPR3c=0.5&w_GPR3b=0.5&w_GPR3a=0.5&w_GPR6=0.5&w_GPRSum=0.5&w_GPR5=0.5&w_GPR4=0.5'
    r = requests.post('http://gttm.jp/gttm_analysis_scripts/ver1_3/GPRanalysis.php', split_params(params, sessionId))
    gpr = r.text[2:]
    f = open(music_xml_path + '.gpr.xml', 'w')
    f.write(gpr)
    f.close()
    gpr = re.sub(r'<supplementary[^>]+/>', '', gpr)
    # gpr = gpr.replace('<GPR xmlns:xlink="http://www.w3.org/1999/xlink">', f'<GPR xmlns:xlink="http://www.w3.org/1999/xlink"><part id="{part_id}">')
    # gpr = gpr.replace('</GPR>', f'</part></GPR>')
    f = open(music_xml_path + '.gpr2.xml', 'w')
    f.write(gpr)
    f.close()
    r = requests.post('http://gttm.jp/gttm_analysis_scripts/ver1_3/submitGPR.php', {'sessionID': sessionId, 'gprData': gpr})
    print('gpr submission', r.text)
    params = 'w_MPR5e=0.5&w_MPR1_rithm_register=0.5&w_MPR5d=0.5&w_MPR5c=0.5&w_MPR5b=0.5&w_MPR5a=0.5&w_MPR1_start_end=0.5&t_MPR4=0.5&t_MPR1=0.5&w_MPR9=0.5&w_MPR8=0.5&w_MPR7=0.5&w_MPR1_length=0.5&w_MPR6=0.5&w_MPR4=0.5&t_MPR5c=0.5&w_MPR3=0.5&t_MPR5b=0.5&w_MPR2=0.5&w_MPR1=0.5&w_MPR10=0.5&t_MPR5a=0.5'
    r = requests.post('http://gttm.jp/gttm_analysis_scripts/ver1_3/MPRanalysis.php', split_params(params, sessionId))
    mpr = r.text[2:]
    f = open(music_xml_path + '.mpr.xml', 'w')
    f.write(mpr)
    f.close()
    return mpr


def get_data():
    for i in range(100):
        try:
            get_mpr(R"output\gttm\RM-P%03d.SMF_SYNC.MID.xml" % (i + 1))
        except:
            print('error')

def evaluate(file, subbeat_count=4):
    try:
        print('Processing ' + file)
        midi = pretty_midi.PrettyMIDI(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file))
    except:
        print('Error loading ' + file)
        return
    rolls, _ = get_rolls(midi, subbeat_count=subbeat_count, drums=0, melody=1, others=0)
    original_length = rolls.shape[1]
    n_subbeat, downbeat_bins, boundaries, subbeat_time = prepare_quantization(midi, subbeat_count)
    xml_file = os.path.join('output', 'gttm', os.path.basename(file) + '.xml.mpr.xml')
    root = xml.etree.ElementTree.parse(xml_file).getroot()
    predicted_downbeat_bins = [int(round(float(node.get('at')) * subbeat_count)) for node in root.findall('metric') if int(node.get('dot')) >= 5]
    predicted_downbeat_bins = np.array(predicted_downbeat_bins)
    downbeat_gt = np.zeros(original_length // subbeat_count)
    downbeat_gt[downbeat_bins[downbeat_bins < original_length] // subbeat_count] = 1
    downbeat_pred = np.zeros(original_length // subbeat_count)
    downbeat_pred[np.round(predicted_downbeat_bins[predicted_downbeat_bins < original_length] / subbeat_count + 1e-3).astype(int)] = 1
    return eval_beat_result(downbeat_pred, downbeat_gt)

if __name__ == '__main__':
    f = open('data/rwc_downbeat_eval_indices.txt', 'r')
    lines = [line.strip() for line in f.readlines() if line.strip() != '']
    f.close()
    results = []
    for line in lines:
        result = evaluate(line)
        print(result)
        results.append(result)
    print(np.mean(results), np.std(results))
    # split_files = get_split('rwc_multitrack_hierarchy_v6_supervised', 'test')
    # split_files = ['RM-P%03d.SMF_SYNC.MID' % (i + 1) for i in range(100)]
    # for file in split_files:
    #     pre_quantize(file)

    # get_mpr(R"D:\workplace\GTTMEditorWebStart_1_4\01_Waltz in E flat Grande Valse Brillante Op.18.xml")
