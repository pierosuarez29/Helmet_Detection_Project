import torch
print(torch.cuda.is_available())  # Debe salir True
print(torch.cuda.device_count())  # Debe salir >= 1
print(torch.cuda.get_device_name(0))  # Nombre de tu GPU
