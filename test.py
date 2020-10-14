#!/usr/bin/env python3

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
import torchaudio

from model import WaveGrad
from data import AudioDataset, MelSpectrogramFixed
from benchmark import estimate_average_rtf_on_filelist, iters_schedule_grid_search
from utils import ConfigWrapper, load_latest_checkpoint, show_message


def run(config, args):
    show_message('Loading model...', verbose=args.verbose)
    model = WaveGrad(config).cuda()
    show_message(f'Number of parameters: {model.nparams}', verbose=args.verbose)
    model, _, iteration = load_latest_checkpoint(
        args.expdir, model, itr=config.test_config.ckpt_num)

    test_dataset = AudioDataset(config, training=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.test_config.batch_size)
    mel_fn = MelSpectrogramFixed(
        sample_rate=config.data_config.sample_rate,
        n_fft=config.data_config.n_fft,
        win_length=config.data_config.win_length,
        hop_length=config.data_config.hop_length,
        f_min=config.data_config.f_min,
        f_max=config.data_config.f_max,
        n_mels=config.data_config.n_mels,
        window_fn=torch.hann_window
    ).cuda()

    if config.test_config.linear:
        schedule = {
            'init': torch.linspace,
            'init_kwargs': {
                'steps': config.test_config.n_iter,
                'start': config.training_config.test_noise_schedule.betas_range[0],
                'end': config.training_config.test_noise_schedule.betas_range[1]
            }
        }
    else:
        outdir = os.path.join(args.expdir, 'schedules')
        os.makedirs(outdir, exist_ok=True)
        best_schedule_path = os.path.join(
            outdir,
            'model' + str(iteration) + '_' +
            'iters' + str(config.test_config.n_iter) + '_'
            + 'best_schedule.pt')
        if os.path.exists(best_schedule_path):
            iters_best_schedule = torch.load(best_schedule_path)
        else:
            iters_best_schedule, _ = iters_schedule_grid_search(
                model=model,
                n_iter=config.test_config.n_iter,
                config=config,
                step=config.test_config.grid_step,
                test_batch_size=config.test_config.grid_batch_size,
                path_to_store_stats=os.path.join(
                    outdir, 'grid_search_stats.pt'),
                verbose=args.verbose
            )
            torch.save(iters_best_schedule, best_schedule_path)
        schedule = {'init': lambda **kwargs: torch.FloatTensor(iters_best_schedule),
                    'init_kwargs': {'steps': config.test_config.n_iter}}

    model.set_new_noise_schedule(
        init=schedule['init'],
        init_kwargs=schedule['init_kwargs']
    )

    outdir = os.path.join(args.expdir, 'wav')
    os.makedirs(outdir, exist_ok=True)
    for test_sample in test_dataloader:
        test_data, test_path = test_sample
        test_path = test_path[0]
        show_message('Processing ' + test_path + '...', verbose=args.verbose)
        mel = mel_fn(test_data.cuda())
        outputs = model.forward(
            mel, store_intermediate_states=False
        )
        out_path = os.path.join(outdir, os.path.basename(test_path))
        torchaudio.save(out_path, outputs.squeeze().cpu(),
                        config.data_config.sample_rate)

    if config.test_config.calc_rtf:
        rtf_stats = estimate_average_rtf_on_filelist(
            config.training_config.test_filelist_path,
            config, model, verbose=args.verbose)
        print(rtf_stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str)
    parser.add_argument('-e', '--expdir', required=True, type=str)
    parser.add_argument('-v', '--verbose', required=False, default=True, type=bool)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    run(config, args)
