import os

dir = 'w_dirs/depthformer/2024_02_28_09_13'
nam = '20240228_091309.log.json'

#dir = 'w_dirs/depthformer/2024_02_14_08_44'
#nam = '20240214_084512.log.json'

fname = os.path.join(dir, nam)

val_lines = []
with open(fname) as f:
    lines = f.readlines()
    lines = lines[1:]

    for idx, l in enumerate(lines):
        #if idx % 33 == 32:
        #    val_lines.append(l)
        if l.startswith('{"mode": "val"'):
            val_lines.append(l)

dname = os.path.join(dir, 'val_filtered.txt')
df = open(dname, 'a')
for l in val_lines:
    df.write(l)
df.close()



