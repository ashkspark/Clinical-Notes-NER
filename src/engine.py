from tqdm import tqdm

#################### TRAIN ####################
def train_one_step(model, data, optimizer, device):
    optimizer.zero_grad()
    
    for k, v in data.items():
        data[k] = v.to(device)
        
    output, loss = model(**data)
    output = output.cpu().detach().numpy().tolist()

    loss.backward()
    optimizer.step()
    
    return output, float(loss)

def train_one_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    
    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        output, loss = train_one_step(model, data, optimizer, device)
        total_loss += loss
        
    scheduler.step()
        
    return total_loss