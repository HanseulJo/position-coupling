import time
import torch

from src.model import DECODER_BASED, ENCODER_DECODER_BASED, CustomDecoderOnlyT5


def print_training_update(phase,
                          device,
                          epoch,
                          step,
                          lr,
                          loss,
                          start_time):
    update_data = [
        phase.upper(), f'Device={device}', 
        f'Epoch={epoch}' if epoch is not None else None,
        f'Step={step}' if step is not None else None,
        f'LR={lr:.7f}' if lr is not None else None,
        f'Loss={loss:.5f}' if loss is not None else None,
        f'PassedTime={time.time() - start_time:.3f}s' if start_time is not None else None
    ]
    print('|', ' '.join(item for item in update_data if item), flush=True)

@torch.no_grad()
def print_example(cfg, ctx, epoch, phase, tokenizer, dataset, model:CustomDecoderOnlyT5, verbose=True):
    device = model.device
    data = dataset[phase][0]
    input_str, label_str = data[:2]
    input_positions, label_positions = None, None
    if len(data) > 2:
        input_positions, label_positions = data[2:]
    label_str = label_str.replace(" ", '')

    if cfg.model.model_name in DECODER_BASED and cfg.task.eos:
        input_ids = torch.LongTensor([enc.ids[:-1] for enc in tokenizer.encode_batch([f"{input_str}="])]).to(device)  # delete [EOS]
    elif cfg.model.model_name in ENCODER_DECODER_BASED+DECODER_BASED:
        input_ids = torch.LongTensor([enc.ids for enc in tokenizer.encode_batch([input_str])]).to(device)
    else:
        raise ValueError(f"model_name: {cfg.model.model_name}")
    
    max_length = 250
    position_ids = None
    if input_positions is not None:
        position_ids = [0] + input_positions + label_positions
        position_ids += [0] * (max_length - len(position_ids))
        position_ids = torch.LongTensor(position_ids).to(device)

    model_input = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'max_length': max_length
    }
    with ctx:
        model_output = model.generate(**model_input)
    decoded = tokenizer.decode_batch(model_output.cpu().tolist())[0]
    decoded = decoded.replace(' ', '')

    if cfg.model.model_name in ENCODER_DECODER_BASED:
        text = f"Epoch {epoch} : {input_str}={decoded} v.s. {label_str} (label)"
    elif cfg.model.model_name in DECODER_BASED:
        text = f"Epoch {epoch} : {decoded} v.s. {label_str} (label)"
    else:
        raise ValueError(f"model_name: {cfg.model.model_name}")
    
    if verbose:
        print("Epoch", epoch, 
            "\nInput  :", input_str, 
            "\nLabel  :", label_str, 
            "\nModelOutput:", model_output, 
            "\nDecoded:", decoded)

    return input_str, label_str, decoded, text

def print_2D(model_inputs, pred, id_0):
    position_ids = model_inputs['position_ids'][:,0].cpu().numpy()
    print("Position  :", position_ids)
    width = position_ids[0][position_ids[0]>0].max() - position_ids[0][position_ids[0]>0].min()
    height = position_ids[1][position_ids[1]>0].max() - position_ids[1][position_ids[1]>0].min()
    
    lab = model_inputs['labels'][0, 1:]
    prediction = pred[0, :-1][lab != -100].cpu().numpy()[:-1].reshape(height, width) - id_0
    label = lab[lab != -100].cpu().numpy()[:-1].reshape(height, width) - id_0

    print("Prediction:")
    print(prediction)
    print("Label:")
    print(label)
