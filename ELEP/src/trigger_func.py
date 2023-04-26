import numpy as np
from obspy.signal.trigger import trigger_onset

def detection_summary_simple(data, thrd=0.1):
    # exclude the first 5 seconds to avoid artifacts
    data[:300] = 0.

    triggers = trigger_onset(data, thrd, thrd / 2)
    detections = []
    for s0, s1 in triggers:
        detection = [s0, s1]
        detections.append(detection)
        
    return detections

def picks_summary_simple(data, thrd=0.1):
    # exclude the first 5 seconds to avoid artifacts
    data[:300] = 0.

    triggers = trigger_onset(data, thrd, thrd / 2)
    picks = []
    for s0, s1 in triggers:
        s_peak = s0 + np.argmax(data[s0 : s1 + 1])

        if s_peak is not None:
            picks.append(s_peak)
    
    return picks