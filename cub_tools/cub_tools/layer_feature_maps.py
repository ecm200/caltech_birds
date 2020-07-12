import torch
import numpy as np

def extract_feature_maps(model, dataloader, fc_feature_extractions, device, batch_limit=None):
    was_training = model.training
    model.eval()

    img_paths = []
    print('Commencing predictions and feature extraction minibatch..', end='')
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(dataloader):
            if i % 25 == 0:
                print('{}..'.format(i), end='')
            
            # Allow early termination of batches if specified
            if i >= batch_limit:
                print('..exiting at specified batch limit ({})'.format(batch_limit))
                break

            inputs = inputs.to(device)
            labels = labels.to(device)


            out = model(inputs)
            _, preds = torch.max(out, 1)

            if i == 0:
                labels_truth = labels.cpu().numpy()
                labels_pred = preds.cpu().numpy()
                for path in paths:
                    img_paths.append(path)
            else:
                labels_truth = np.concatenate((labels_truth,labels.cpu().numpy()))
                labels_pred = np.concatenate((labels_pred,preds.cpu().numpy()))
                for path in paths:
                    img_paths.append(path)

            fc_feature_extractions[i] = fc_feature_extractions[i].cpu().numpy()

    return {'labels truth' : labels_truth, 
            'labels pred' : labels_pred,
            'image paths' : img_paths,
            'feature extractions' : fc_feature_extractions}