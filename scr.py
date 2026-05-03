with open("lines.txt", "r") as lines:
    for line in lines:
        if "]" not in line[-7:] and line[-7:] != "\n":
            print(line[-7:-1]+", ", end="")