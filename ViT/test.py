import time
import torch
import torch.nn as nn

device = torch.device('cuda')
loss_func = nn.CrossEntropyLoss()

def test(model,
         test_data,
         device,
         loss_func):
    start = time.time()
    model.eval()
    with torch.no_grad():
        tbatch_loss, tbatch_acc = 0, 0
        for tbatch, (test_images, test_labels) in enumerate(test_data):
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            
            test_outputs = model(test_images)
            test_loss = loss_func(test_outputs, test_labels)
            output_index = torch.argmax(test_outputs, dim=1)
            test_acc = (output_index==test_labels).sum()/len(test_outputs)
            
            tbatch_loss += test_loss.item()
            tbatch_acc += test_acc.item()
            
            del test_images; del test_labels; del test_outputs
            torch.cuda.empty_cache()
        end = time.time()

    print(f'Test Loss: {tbatch_loss/(tbatch+1):.3f},' 
          f' Test Acc: {tbatch_acc/(tbatch+1):.3f},'
          f' Time: {end-start:3f}s')