import numpy as np
import math

p_start = np.array([0, 0])
p_dest = np.array([0, 10])
w = np.array([-5, 0])
speed = 5

v = p_dest - p_start
v_norm = np.linalg.norm(v)
v_unit = v / v_norm
u_drone = speed * v_unit - w
u_total = u_drone + w
time = v_norm / u_total[1]
energy = np.linalg.norm(u_drone)**2 * time

# Tính góc lệch giữa u_drone (hướng drone cần di chuyển) và v_unit (hướng đích)
dot = u_drone[0] * v_unit[0] + u_drone[1] * v_unit[1]
norm_u_drone = np.linalg.norm(u_drone)
norm_v_unit = np.linalg.norm(v_unit)
cos_theta = dot / (norm_u_drone * norm_v_unit)
cos_theta = min(1.0, max(-1.0, cos_theta))
theta = math.degrees(math.acos(cos_theta))

print("Hướng từ đầu đến đích:", v)
print("Hướng đi thực tế:", u_total)
print("Hướng drone cần di chuyển:", u_drone)
print("Hướng lệch:", f"{theta:.2f} độ")
print("Năng lượng:", energy)
print("Đề xuất:", "Tăng tốc độ (lệch hướng)" if theta > 10 else "Giảm tốc độ (năng lượng cao)" if energy > 100 else "Tốc độ hợp lý")Lưu Anh Tuấn Huy_baitap_vector
