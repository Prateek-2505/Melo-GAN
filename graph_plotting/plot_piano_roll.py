import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pretty_midi

def plot_piano_roll(notes, title="Generated Sequence"):
    """
    Plots a Piano Roll from (N, 4) tensor: [pitch, velocity, duration, start_time].
    """
    # Unpack mock data (In reality, you'd parse the GAN output tensor)
    # start_times = np.cumsum(np.random.uniform(0.2, 0.5, 20))
    # pitches = np.random.randint(60, 85, 20)
    # durations = np.random.uniform(0.2, 0.4, 20)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Draw Notes as rectangles
    for t, p, d in zip(start_times, pitches, durations):
        # Rectangle(xy, width, height)
        rect = patches.Rectangle((t, p), d, 1.0, linewidth=1, edgecolor='black', facecolor='orange')
        ax.add_patch(rect)
        
    ax.set_title(f'Piano Roll: {title}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MIDI Pitch')
    ax.set_ylim(35, 95) # Piano range
    ax.set_xlim(0, max(start_times) + 1)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig('piano_roll_'+title+'.png')
    plt.close()
    

if __name__ == "__main__":
    

    angry = os.path.join("eval_temp_midi", "eval_Angry_0.mid")
    happy= os.path.join("eval_temp_midi", "eval_Happy_0.mid")
    sad = os.path.join("eval_temp_midi", "eval_Sad_0.mid")
    calm = os.path.join("eval_temp_midi", "eval_Calm_0.mid")
    files = [angry,happy,sad,calm]
    if len(files)==0:
        print(f"No files generated found ")
    for fp in files:
        pm = pretty_midi.PrettyMIDI(fp)
        pitches = []
        velocities = []
        start_times = []
        durations = []
        for inst in pm.instruments:
            for n in inst.notes:
                pitches.append(n.pitch)
                velocities.append(n.velocity)
                start_times.append(n.start)
                durations.append(n.end - n.start)

        if len(pitches) == 0:
            print(f"No notes found in {fp}, skipping.")
            continue

        pitches = np.array(pitches)
        velocities = np.array(velocities)
        start_times = np.array(start_times)
        durations = np.array(durations)

        # Build notes array as [pitch, velocity, duration, start_time]
        notes = np.column_stack([pitches, velocities, durations, start_times])

        # Ensure plot_piano_roll (which expects module-level names in this file) can access arrays
        globals().update({
            "pitches": pitches,
            "durations": durations,
            "start_times": start_times
        })

        title = os.path.splitext(os.path.basename(fp))[0]
        plot_piano_roll(notes, title=title)
        print(f"Plotted {fp}")
