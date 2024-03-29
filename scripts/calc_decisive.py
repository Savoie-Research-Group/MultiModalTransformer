#write the decisive string to the .txt file R/M/I/N represents decisive on certain token F represents non-decisive

from rdkit import Chem

def write_decisive(file_path,dec_string):
    with open(file_path,'a') as f:
        f.write(dec_string+'\n')
        f.close()
    return None
def calc_decisive(logger,batch, model, src_maxlen, tgt_maxlen, idx_to_ch_dict, target_start_token_idx, target_end_token_idx, mol_path, r_path, ms_path, ir_path, nmr_path):
    count_top1 = 0
    count_top5 = 0
    beam_size = 1
    source, target = batch
    bs = source.size(dim=0)
    preds,r_list,ms_list,ir_list,nmr_list = model.beam_search(source, beam_size, src_maxlen, tgt_maxlen, target_start_token_idx, target_end_token_idx)
    logger.info(f'beam search complete')
    for i in range(bs):
        target_text = "".join([idx_to_ch_dict[int(_)] for _ in target[i, :]])
        target_text = target_text.replace("PAD_WORD","")
        target_text = target_text.replace("<","")
        target_text = target_text.replace("$","")
        preds[i].sort(key = lambda x:x[1])
        pred_dict = {}
        for j in range(beam_size):
            pred_dict["prediction_{}".format(j)] = ""
            for idx in preds[i][-1-j][0]:
                pred_dict["prediction_{}".format(j)] += idx_to_ch_dict[idx]
                if idx == target_end_token_idx:
                    break
            pred_dict["prediction_{}".format(j)] = pred_dict["prediction_{}".format(j)].replace("<","")
            pred_dict["prediction_{}".format(j)] = pred_dict["prediction_{}".format(j)].replace("$","")
            try:
                pred_dict["prediction_{}".format(j)] = Chem.CanonSmiles(pred_dict["prediction_{}".format(j)])
            except:
                pred_dict["prediction_{}".format(j)] = "XXX"

        prediction_top1 = pred_dict["prediction_0"]
        with open(mol_path,'a') as f:
            f.write(target_text+';'+prediction_top1+'\n')
            f.close()
        write_decisive(r_path,r_list[i])
        write_decisive(ms_path,ms_list[i])
        write_decisive(ir_path,ir_list[i])
        write_decisive(nmr_path,nmr_list[i])
    return None

