

# 有用的Issue：
关于协方差的设置讨论：
https://github.com/hku-mars/Point-LIO/issues/37

[imu作为输入与作为输出的区别？](https://github.com/hku-mars/Point-LIO/issues/23)

# imu_en 和 Use_imu_as_input 的区别？？

好像`imu_en`设置为false后就完全没有IMU的事情了。即完全使用LiDAR做所有的状态估计。此时 `use_imu_as_input` 应该是没有任何作用。根据官方readme，需要将`use_imu_as_input`设置为0.

而`use_imu_as_input`是IMU到底用作输入，还是状态估计的控制。必须`imu_en`为true。

`imu_en`影响的地方：
1. IMU重力align时，无IMU就采用给定值进行计算；
2. 第一帧lidar/后续lidar时，获取加速度和角速度；之后：在ekfom观测模型(imu_ouput)进行计算，同时包括了饱和判断；





# 关于Input和Output

默认状态维度：
论文(output)：R, p, v, g, bg, ba, (w, a)
代码：多2个外参，
MTK_BUILD_MANIFOLD(state_input,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((vect3, gravity))
);

MTK_BUILD_MANIFOLD(state_output,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, omg))
((vect3, acc))
((vect3, gravity))
((vect3, bg))
((vect3, ba))
);

Input下：
IMU的加速度/角速度不是状态量，和普通的SLAM相同。
（6论文量+2外参）*3 = 24维

Output下：
加速度、角速度是需要估计的状态量，因此可以应对饱和的问题，即估计的角速度/加速度可以超过阈值。
（8论文量+2外参）*3 = 30维


