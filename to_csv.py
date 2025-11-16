import os
import csv

folder = 'ICBHI_final_database'
csv_filename = 'icbhi_segments.csv'
headers = ['filename', 'start_time', 'end_time', 'crackle', 'wheeze']

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    for fname in os.listdir(folder):
        if fname.endswith('.txt'):
            wav_file = fname.replace('.txt', '.wav')
            with open(os.path.join(folder, fname)) as ann_file:
                for line in ann_file:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        row = [wav_file] + parts
                        writer.writerow(row)
