import numpy as np
import pretty_midi
import sys

roll = np.load(sys.argv[1])

pm = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # piano

for row in roll:
    pitch = int(np.clip(row[0], 0, 127))
    velocity = int(max(1, min(127, row[1])))
    duration = max(0.05, float(row[2]))
    start = max(0.0, float(row[3]))
    end = start + duration
    instrument.notes.append(pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=start,
        end=end
    ))

pm.instruments.append(instrument)
pm.write("generated_sample.mid")
print("Wrote generated_sample.mid")
