import os

def parse_config(args):
    if os.path.isfile(args.config) is False:
        print('Fatal error: No config file.')
        exit()
    keywords = ['src_test','tgt_test','null_src_test','null_tgt_test','mode_lst','best_chk_path','ch_to_idx_dict_path','idx_to_ch_dict_path','real_test_output_path','null_test_output_path','noise_level','return_spec']
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
