import time

from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops.composite as C
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype

from evaluation import Metrics


class CrossEntropyLoss(nn.Cell):
    """ Normal cross entropy loss with `ignore_index`. Note that any negative label values 
    do not contribute to the loss calculation. """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.epsilon = Tensor(1e-5, mstype.float32)
        self.softmax = P.Softmax()
        self.log = P.Log()
        self.neg = P.Neg()
        self.sum = P.ReduceSum()
        self.argmax = P.Argmax(axis=2, output_type=mstype.int32)
    
    def construct(self, logits, labels):
        """ Calculate the average cross entropy loss, with padded labels excluded. 
        Args:
            logits: Tensor of shape (B, qlen, C)
            labels: Tensor shape (B, qlen)
        Return:
            Average cross entropy loss, predictions of shape (B, qlen, C)
        """
        labels = F.cast(labels, mstype.int32)
        logits = F.cast(logits, mstype.float32)
        onehot_labels = self.onehot(labels, logits.shape[2], self.on_value, self.off_value)
        probs = self.softmax(logits)  # (B, qlen, C)

        cross_entropy = self.neg(onehot_labels * self.log(probs))
        cross_entropy = self.sum(cross_entropy, 2)  # (B, qlen)
        num_labels = self.sum(onehot_labels) + self.epsilon
        avg_cross_entropy = self.sum(cross_entropy) / num_labels
        predictions = self.argmax(probs)

        return avg_cross_entropy, predictions


class NetworkWithGrossEntropyLoss(nn.Cell):
    """ Wrapper cell for loss calculation. """
    def __init__(self, backbone):
        super(NetworkWithGrossEntropyLoss, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = CrossEntropyLoss()
    
    def construct(self, features, labels):
        logits = self.backbone(F.cast(features, mstype.float32))
        loss, predictions = self.loss_fn(logits, labels)
        return loss, predictions


class TrainOneStepCell(nn.Cell):
    """ Wrapper cell for gradients calculation and clipping. """
    def __init__(self, loss_network, optimizer, max_grad_norm=5.0, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.loss_network = loss_network
        self.loss_network.set_grad()
        self.loss_network.set_train()
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
    
    def construct(self, *inputs):
        weights = self.optimizer.parameters
        loss, predictions = self.loss_network(*inputs)
        sens = (
            F.fill(F.dtype(loss), F.shape(loss), self.sens),
            F.fill(F.dtype(predictions), F.shape(predictions), self.sens),
        )

        grad_fn = self.grad(self.loss_network, weights)
        grads = grad_fn(*inputs, sens)

        retval = (loss, predictions)
        grads = C.clip_by_global_norm(grads, self.max_grad_norm)
        succ = self.optimizer(grads)
        return F.depend(retval, succ)


def run_mindspore_model_one_epoch(model, dataloader, metrics: Metrics):
    for data in dataloader.create_dict_iterator():
        features = data["features"]
        labels = data["labels"]

        loss, predictions = model(features, labels)
        metrics.update(
            loss=loss.asnumpy().item(),
            labels=labels.asnumpy(),
            predictions=predictions.asnumpy(),
        )
    
    return metrics.evaluate()


def train_mindspore_model(
    model, 
    dataloader, 
    device, 
    epochs, 
    learning_rate=1e-3, 
    max_grad_norm=5.0, 
    padding_label=-1,
):
    if padding_label >= 0:
        raise ValueError("For Mindspore, `padding_label` must be negative.")

    optimizer = nn.Adam(
        model.trainable_params(),
        learning_rate=learning_rate,
    )
    
    loss_network = NetworkWithGrossEntropyLoss(model)
    train_network = TrainOneStepCell(loss_network, optimizer, max_grad_norm)
    metrics = Metrics(padding_label)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        loss, accuracy, fscore = run_mindspore_model_one_epoch(
            train_network, 
            dataloader, 
            metrics,
        )
        print(
            f"Epoch {epoch}: Loss: {loss}, Acc: {accuracy}, F1 Score: {fscore}, "
            f"Total Time: {round(time.time() - epoch_start, 2)} s"
        )


def test_mindspore_model(model, dataloader, device, padding_label=-1):
    if padding_label >= 0:
        raise ValueError("For Mindspore, `padding_label` must be negative.")
    
    loss_network = NetworkWithGrossEntropyLoss(model)
    loss_network.set_grad(False)
    loss_network.set_train(False)
    metrics = Metrics(padding_label)

    epoch_start = time.time()
    loss, accuracy, fscore = run_mindspore_model_one_epoch(
        loss_network,
        dataloader,
        metrics,
    )
    print(
        f"Test: Loss: {loss}, Acc: {accuracy}, F1 Score: {fscore}, "
        f"Total Time: {round(time.time() - epoch_start, 2)} s"
    )
