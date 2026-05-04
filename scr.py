txt_filename = 'lines.txt'

with open(txt_filename, "r") as lines:
    for line in lines:
        if "]" not in line[-7:] and line[-7:] != "\n" and len(line) > 15:
            print(line[-7:-1]+", ", end="")

print("\n-------------------------------------------------")

with open(txt_filename, "r") as lines:
    for line in lines:
        if "]" not in line and line != "\n" and len(line) > 15:
            print(line[15:22]+", ", end="")