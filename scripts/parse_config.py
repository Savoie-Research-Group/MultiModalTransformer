import os

def parse_config(args):
    if os.path.isfile(args.config) is False:
        print('Fatal error: No config file.')
        exit()
    keywords = ['src_train','tgt_train','src_val','tgt_val','null_src_train','null_tgt_train','null_src_val','null_tgt_val','loss_path', 'mode_lst', 'best_chk_path', 'end_chk_path']
    keywords = [_.lower() for _ in keywords]
    list_delimiters = [',']
    space_delimiters = ['&']
    configs = {i:None for i in keywords}
    with open(args.config, 'r') as f:
        for line in f:
            fields = line.split()
            if '#' in fields: del fields[fields.index('#'):]
            for i in keywords:
                if i in fields:
                    ind = fields.index(i)+1
                    if len(fields) >= ind+1:
                        configs[i] = fields[ind]
                        for j in space_delimiters:
                            if j in configs[i]:
                                configs[i] = " ".join([_ for _ in configs[i].split(j)])
                        for j in list_delimiters:
                            if j in configs[i]:
                                configs[i] = configs[i].split(j)
                                break
    return configs
