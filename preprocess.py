import sys

f = open(sys.argv[1])

o = open(sys.argv[2], "w")

for line in f:
    line = line[:-1]
    if line.startswith("NONE") or line.startswith("CP_START") or line.startswith("CP") or line.startswith(
            "CC") or line.startswith("OTHERS"):
        continue
    else:
        o.write(line+"\n")
o.close()
