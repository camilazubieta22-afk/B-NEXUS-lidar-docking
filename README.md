# Docking autónomo del iRobot Create 3 usando solo LiDAR — Kalman Dock Challenge, HRFEST 2026.
**Autora:** Camila Zubieta

**Video demo:** https://youtu.be/T5m1wu69nyY?si=C_546L7OoQ4B9G1t

## Idea
El escaneo LiDAR se transforma al sistema de coordenadas `base_link` mediante TF, corrigiendo la orientación de 180° y el desplazamiento del sensor respecto al centro del robot. Luego, se utiliza RANSAC para extraer las paredes como rectas y detectar sobre ellas dos segmentos de 8 cm que sobresalen 8 cm, separados por un hueco de 9,5 cm. El centro de ese hueco y la normal de la pared permiten estimar el eje del dock. Esta estimación se filtra en el sistema de coordenadas `odom` mediante un filtro exponencial y el rechazo de valores atípicos. Finalmente, el controlador sigue un punto objetivo situado sobre el eje del dock y corrige simultáneamente los errores lateral y angular para lograr un acoplamiento preciso.

Estados: `SEARCH → EXPLORE → GOTO_PREDOCK → APPROACH → FINAL → DOCKED`, con `BACKOFF` para reintentar.

## Estructura
* `dock_lidar/perception.py`: transforma el escaneo LiDAR en información geométrica, detectando paredes, cajas y el eje del dock.
* `dock_lidar/tracker.py`: filtra y estabiliza las estimaciones del dock a lo largo del tiempo.
* `dock_lidar/controller.py`: implementa la máquina de estados y el control de movimiento del robot.
* `dock_lidar/dock_node.py`: implementa el nodo ROS 2 y publica marcadores para visualizar las detecciones en RViz.
* `config/params.yaml`: contiene los parámetros de configuración del sistema.
* `tools/sim2d.py`: implementa un simulador 2D para realizar pruebas offline del algoritmo.

## Interfaces
- **Usa:** `/scan`, `/cmd_vel`, `/tf`, `/tf_static` y `/dock_status` (solo `is_docked`, para detenerse).
- **No usa:** la acción `/dock`, `/ir_*`, `dock_visible`, `/sim_ground_truth_*`, `/gazebo/*` ni `std_dock_link`.
- **Sin coordenadas del mundo:** solo las medidas del marcador (8 cm y 9.5 cm).

## Ejecución
```bash
cd ~/sim_ws && colcon build --symlink-install --packages-select dock_lidar
source install/setup.bash

# Terminal 1
ros2 launch create3_dock_challenge challenge_world.launch.py x:=-2.0 y:=1.0 yaw:=3.0

# Terminal 2
ros2 launch dock_lidar dock.launch.py
```

En el robot real: `ros2 launch dock_lidar dock.launch.py use_sim_time:=false`

## Resultados (Gazebo, `v_max = 0.20`)

| Pose inicial (x, y, yaw) | Tiempo | Error lateral | Error angular |
|---|---|---|---|
| (0.41, −0.18, 0.36) por defecto | 10.0 s | 0.2 mm | 0.59° |
| (0.9, 0.7, −1.2) | 11.3 s | −0.1 mm | 0.09° |
| (1.2, −1.2, 2.5) | 28.5 s | −0.4 mm | −0.18° |
| **(−2.0, 1.0, 3.0) lejos y de espaldas — video** | 25.8 s | −0.6 mm | −0.26° |

Todos los intentos acoplaron, sin reintentos ni choques.

Con `v_max = 0.28` la pose (−2.0, 1.0, 3.0) no acopló: entró en un bucle de 5 reintentos en `BACKOFF`. Por eso se mantiene `v_max = 0.20`.
