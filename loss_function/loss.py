from loss_function.uda_loss import uda_loss

def get_loss_function(loss_name):
  if loss_name == "uda":
    return uda_loss