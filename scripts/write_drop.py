def write_drop(input_list,name):
    for i in range(len(input_list)):
        with open(name,"a") as f:
            f.write(str(float(input_list[i])))
            f.write("\n")
            f.close()
    return None
