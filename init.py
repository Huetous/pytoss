from toss.hooks import Hook
from toss.layers import ConvLSUVLayer


def LSUV(model, trainer):
    def lsuv_init(m, xb):
        h = Hook(m, append_stat)
        while model(xb) is not None and abs(h.mean) > 1e-3:
            m.bias -= h.mean
        while model(xb) is not None and abs(h.std - 1) > 1e-3:
            m.weight.data /= h.std
        h.remove()

    def get_batch(gen, trainer):
        trainer.x_batch, trainer.y_batch = next(iter(gen))
        for cb in trainer.cbs:
            cb.set_trainer(trainer)
        trainer("begin_batch")
        return trainer.x_batch, trainer.y_batch

    def append_stat(hook, mod, inp, out):
        d = out.data
        hook.mean = d.mean().item()
        hook.std = d.std().item()

    def find_modules(m, cond):
        if cond(m):
            return [m]
        return sum([find_modules(o, cond) for o in m.children()], [])

    modules = find_modules(trainer.model, lambda o: isinstance(o, ConvLSUVLayer))
    xb, yb = get_batch(trainer.data.train_gen, trainer)
    model.cuda()

    for m in modules:
        lsuv_init(m, xb)

    trainer.model = model
    return model, trainer
