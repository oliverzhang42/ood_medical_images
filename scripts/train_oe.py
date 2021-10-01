import numpy as np
import os
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from scripts.utils import create_dataloader, get_checkpoint, save_results
from src.losses import get_loss_outlier_exposure
from src.metrics import calc_metrics
from src.network import ResNet
from src.utils import DATASET_PATHS, NUM_CLASSES, evaluate
from tqdm import tqdm


def main(config_path):
    config = OmegaConf.load(config_path)
    config['root'] = DATASET_PATHS
    config['num_classes'] = NUM_CLASSES[config.iid_name]

    # Creates the dataloaders
    test_loader = create_dataloader(config, config.iid_name, 'test')
    train_loader = create_dataloader(config, config.iid_name, 'train')
    aux_loader = create_dataloader(config, 'skeletal-age', 'train')
    ood_loaders = [create_dataloader(config, ood_name, 'test') for ood_name in config.ood_names]

    # Creates the experiment folder
    experiment_folder = f'checkpoints/{config.experiment_name}'
    os.makedirs(experiment_folder, exist_ok=True)

    # Performance
    performance = [{} for i in range(config.num_models)]

    # For each model in our ensemble
    for i in range(config.num_models):
        
        # Setting things up to begin training
        net = ResNet(config.num_classes).cuda()
        optimizer = optim.Adam(net.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, factor=0.8)
        checkpoints_folder = f'checkpoints/{config.experiment_name}/{i}'
        os.makedirs(checkpoints_folder, exist_ok=True)

        # Creates the metrics dictionary
        ood_metrics = {}
        for ood_name in config.ood_names:
            ood_metrics[ood_name] = {'fpr_at_95_tpr': [], 'detection_error': [], 'auroc': []}
        ood_metrics['test_accuracy'] = []

        # For Early Stopping
        early_stop = 0
        best_value = 0

        # Beginning the Training Process!
        for epoch in range(config.num_epochs):
            # Train phase
            net.train()
            print(f'Starting epoch {epoch}!')
            pbar = tqdm(total=len(train_loader))
            for train_iter, (in_sample, out_sample) in enumerate(zip(train_loader, aux_loader), 0):
                optimizer.zero_grad()
                images, labels = in_sample
                ood_images, ood_labels = out_sample
                batch_images = torch.cat((images, ood_images))
                batch_logits, _ = net(batch_images.cuda())
                logits, ood_logits = batch_logits[:len(images)], batch_logits[len(images):]
                total_loss = get_loss_outlier_exposure(
                    logits,  
                    ood_logits,
                    labels.cuda(),
                    beta=config.beta
                )
                total_loss.backward()
                optimizer.step()
                pbar.update()
            pbar.close()

            # Eval phase
            if epoch + 1 >= config.eval_start:
                net.eval()

                # Train evaluation
                train_pred, _ = evaluate(train_loader, net, config.mode)
                train_accuracy = round(float(np.mean(train_pred)), 4)
                print(f'Training accuracy {train_accuracy}')

                # Test evaluation
                pred, confidences = evaluate(test_loader, net, config.mode)
                labels = np.ones(confidences.shape[0])
                test_accuracy = round(float(np.mean(pred)), 4)
                ood_metrics['test_accuracy'].append(test_accuracy)
                print(f'Testing accuracy {test_accuracy}')

                # OOD Evaluation
                for ood_name, ood_loader in zip(config.ood_names, ood_loaders):
                    ood_pred, ood_confidences = evaluate(ood_loader, net, config.mode)
                    ood_labels = np.zeros(ood_confidences.shape[0])
                    total_confidences = np.concatenate((confidences, ood_confidences))
                    total_labels = np.concatenate((labels, ood_labels))
                    fpr, detection, auroc = calc_metrics(total_confidences, total_labels)
                    ood_metrics[ood_name]['fpr_at_95_tpr'].append(fpr)
                    ood_metrics[ood_name]['detection_error'].append(detection)
                    ood_metrics[ood_name]['auroc'].append(auroc)
                
                print('Saving Results...')
                save_results(checkpoints_folder, ood_metrics)

                # Early Stopping
                early_stop += 1
                if train_accuracy > best_value:
                    early_stop = 0
                    best_value = train_accuracy

                    # Overwrite performance
                    performance[i] = ood_metrics
                    
                    # Remove previous checkpoint
                    prev_checkpoint = get_checkpoint(checkpoints_folder)
                    if prev_checkpoint != '':
                        os.remove(prev_checkpoint)

                    # Add new checkpoint
                    torch.save(
                        {
                            'net': net.state_dict(),
                            'config': config,
                        }, f'{checkpoints_folder}/model_{best_value}.pth'
                    )
                    print(f'Early stop beaten. Now best accuracy is {best_value}.')

                scheduler.step(train_accuracy)

            if early_stop == config.early_stop:
                print('early_stop reached')
                break
        
        print(f'Done with model {i}! Moving on... \n')
    
    print('Done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    args = parser.parse_args()
    main(args.config)