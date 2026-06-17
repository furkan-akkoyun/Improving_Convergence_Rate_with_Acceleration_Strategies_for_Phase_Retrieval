"""Direct DeepInv RandomPhaseRetrieval baseline from the original notebook."""
from .common import *

# Direct DeepInv random phase retrieval operator, same style as the main notebook.
deepinv_random_alpha = 4.0
deepinv_random_m = int(deepinv_random_alpha * n_pixels)

physics_deepinv_random = dinv.physics.RandomPhaseRetrieval(
    m=deepinv_random_m,
    img_size=phase_img_size,
    device=device,
)

y_deepinv_random = physics_deepinv_random(x_true)
x_init_deepinv_random = physics_deepinv_random.A_dagger(
    y_deepinv_random,
    n_iter=300,
).clone().detach()

print("DeepInv RandomPhaseRetrieval from main notebook")
print("alpha:", deepinv_random_alpha)
print("m:", deepinv_random_m)
print("x_true shape:", tuple(x_true.shape))
print("y shape:", tuple(y_deepinv_random.shape))
print("x_init shape:", tuple(x_init_deepinv_random.shape))
