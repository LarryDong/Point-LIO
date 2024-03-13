// #include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "Estimator.h"

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
std::vector<int> time_seq;
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
std::vector<V3D> pbody_list;
std::vector<PointVector> Nearest_Points; 
KD_TREE<PointType> ikdtree;
std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
bool   point_selected_surf[100000] = {0};
std::vector<M3D> crossmat_list;
int effct_feat_num = 0;
int k;
int idx;
esekfom::esekf<state_input, 24, input_ikfom> kf_input;
esekfom::esekf<state_output, 30, input_ikfom> kf_output;
state_input state_in;
state_output state_out;
input_ikfom input_in;
V3D angvel_avr, acc_avr;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

Eigen::Matrix<double, 24, 24> process_noise_cov_input()
{
	Eigen::Matrix<double, 24, 24> cov;
	cov.setZero();
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_input>::type cov = MTK::get_cov<process_noise_input>::type::Zero();
	// MTK::setDiagonal<process_noise_input, vect3, 0>(cov, &process_noise_input::ng, gyr_cov_input);// 0.03
	// MTK::setDiagonal<process_noise_input, vect3, 3>(cov, &process_noise_input::na, acc_cov_input); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_input, vect3, 6>(cov, &process_noise_input::nbg, b_gyr_cov); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_input, vect3, 9>(cov, &process_noise_input::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

Eigen::Matrix<double, 30, 30> process_noise_cov_output()
{
	Eigen::Matrix<double, 30, 30> cov;
	cov.setZero();
	cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
	cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
	cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
	cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_output>::type cov = MTK::get_cov<process_noise_output>::type::Zero();
	// MTK::setDiagonal<process_noise_output, vect3, 0>(cov, &process_noise_output::vel, vel_cov);// 0.03
	// MTK::setDiagonal<process_noise_output, vect3, 3>(cov, &process_noise_output::ng, gyr_cov_output); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_output, vect3, 6>(cov, &process_noise_output::na, acc_cov_output); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_output, vect3, 9>(cov, &process_noise_output::nbg, b_gyr_cov);   //0.001 0.05 0.0001/out 0.01
	// MTK::setDiagonal<process_noise_output, vect3, 12>(cov, &process_noise_output::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) =  omega[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}


/* Function:df_dx_output
代码中的状态定义：
	0-((vect3, pos))  
	3-((SO3, rot))
	6-((SO3, offset_R_L_I))
	9-((vect3, offset_T_L_I))
	12-((vect3, vel))
	15-((vect3, omg))
	18-((vect3, acc))
	21-((vect3, gravity))
	((vect3, bg))
	((vect3, ba))
论文中的状态定义：
	R, p, v, bg, ba, g, w, a;
补充：
	代码中，如果估计外参，6和9号位的状态是没有更新的；
	pos，rot，vel，是在IMU的预测部分更新的（get_f_output）
*/
// get_f_output: IMU的状态预测时递推函数，状态转移，对应eq(9)的f(x_k, 0)
Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
	vect3 a_inertial = s.rot * s.acc; 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];				// 第一个位置是pos，即eq(9)中oplus后面的pos增量应该是：pos= velocity*dt
		res(i + 3) = s.omg[i]; 			// 第二个位置是旋转，rotation(3) = omega*dt，而dt在f里面没有出现。
		res(i + 12) = a_inertial[i] + s.gravity[i]; 	// (12)号位是速度，= acc*dt.
	}
	return res;
}

// input模式下，IMU递推/预测时，线性化的F，不包括 角速度、加速度项的递推，因为这两个直接用的IMU的读数。
Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(acc_);
	cov.template block<3, 3>(12, 18) = -s.rot;
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); 
	return cov;
}

//~ This function is deleted by author. Is should be eq(11) F_{w_k}
// 这个函数被作者注释掉了，应该是eq(11)中的 F_{w_k}。但后续实现中，直接加上了，所以没有必要专门再搞个这个函数。
// Eigen::Matrix<double, 24, 12> df_dw_input(state_input &s, const input_ikfom &in)
// {
// 	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
// 	cov.template block<3, 3>(12, 3) = -s.rot.normalized().toRotationMatrix();
// 	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
// 	return cov;
// }


// df_dx_output: IMU的状态预测时，协方差矩阵的转移，对应eq(11)的 F_{x_k} 
// 但这里的 F_x_k 不包含时间间隔dt，和对角线。后续predict时（in `esekfom.hpp`)，会乘上dt，然后加上对角线。
Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
	// 注意下方的块元素，和论文中的位置不一致，因为状态变量的定义顺序不同。按照含义，是可以正确对应的。
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(s.acc);			// d_v/d_rot -> eq(11)'s F31，没有dt
	cov.template block<3, 3>(12, 18) = s.rot;							// d_v/d_acc -> eq(11)'s F38，没有dt

	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();		// d_pos/d_v -> eq(11)'s 中第2行第3列（同样没有dt）
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); 	// d_v/d_g. eq(11)'s 第3行第8列
	cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity(); 		// d_R/d_w; eq(11)'s 第1行第7列
	return cov;
}

// 这个函数，本意应该是：eq(11)中的 F_{w_k}，但是太过于简单，所以没有必要，predict中直接加上了。
// Eigen::Matrix<double, 30, 15> df_dw_output(state_output &s)
// {
// 	Eigen::Matrix<double, 30, 15> cov = Eigen::Matrix<double, 30, 15>::Zero();
// 	cov.template block<3, 3>(12, 0) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(15, 3) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(18, 6) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(24, 9) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(27, 12) = Eigen::Matrix3d::Identity();
// 	return cov;
// }

vect3 SO3ToEuler(const SO3 &rot) 
{
	// Eigen::Matrix<double, 3, 1> _ang;
	// Eigen::Vector4d q_data = orient.coeffs().transpose();
	// //scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	// double sqw = q_data[3]*q_data[3];
	// double sqx = q_data[0]*q_data[0];
	// double sqy = q_data[1]*q_data[1];
	// double sqz = q_data[2]*q_data[2];
	// double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	// double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	// if (test > 0.49999*unit) { // singularity at north pole
	
	// 	_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
	// 	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	// 	vect3 euler_ang(temp, 3);
	// 	return euler_ang;
	// }
	// if (test < -0.49999*unit) { // singularity at south pole
	// 	_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
	// 	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	// 	vect3 euler_ang(temp, 3);
	// 	return euler_ang;
	// }
		
	// _ang <<
	// 		std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
	// 		std::asin (2*test/unit),
	// 		std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	// double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	// vect3 euler_ang(temp, 3);
	// return euler_ang;
	double sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if(!singular)
    {
        x = atan2(rot(2, 1), rot(2, 2));
        y = atan2(-rot(2, 0), sy);   
        z = atan2(rot(1, 0), rot(0, 0));  
    }
    else
    {    
        x = atan2(-rot(1, 2), rot(1, 1));    
        y = atan2(-rot(2, 0), sy);    
        z = 0;
    }
    Eigen::Matrix<double, 3, 1> ang(x, y, z);
    return ang;
}



void h_model_input(state_input &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;

	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		
		{
			auto &points_near = Nearest_Points[idx+j+1];
			
			ikdtree.Nearest_Search(point_world_j, NUM_MATCH_POINTS, points_near, pointSearchSqDis, 2.236); //1.0); //, 3.0); // 2.236;
			
			if ((points_near.size() < NUM_MATCH_POINTS) || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)
				{
					float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);
					
					if (p_body.norm() > match_s * pd2 * pd2)
					{
						point_selected_surf[idx+j+1] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pabcd(3);		// plan's normal value.
						effect_num_k ++;
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}

	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			
			if (extrinsic_est_en)		//~ estimate IMU-LiDAR extrinsics. Default: false.
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);		//~ extrinsics J.
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{   
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			m++;
		}
	}
	effct_feat_num += effect_num_k;
}




// LiDAR的观测模型。
// 计算了：r_I，残差，eq(12)
// 以及：H_L，线性化矩阵（残差对状态偏导），eq(13)
void h_model_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;

	// 计算每个点的最近邻平面
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		{
			auto &points_near = Nearest_Points[idx+j+1];
			
			// 寻找最近邻5点，为什么要设定2.236m的搜索半径？
			ikdtree.Nearest_Search(point_world_j, NUM_MATCH_POINTS, points_near, pointSearchSqDis, 2.236); 
			
			if ((points_near.size() < NUM_MATCH_POINTS) || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)			// 满足平面设定。pabcd中的abcd是平面方程系数
				{
					float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);
					if (p_body.norm() > match_s * pd2 * pd2)
					{
						// point_selected_surf[i] = true;
						point_selected_surf[idx+j+1] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pabcd(3);
						effect_num_k ++;
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}

	ekfom_data.M_Noise = laser_point_cov;
	// h_x 是 eq(13)中的 H_L_{k+1}，但最后18列都是0，所以省略了，在计算 eq(20) 时只用12列，P也只有前12行有用。
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);		
	ekfom_data.z.resize(effect_num_k);	// z是lidar的残差，eq(12) 计算

	// 找到近邻点/平面后，开始残差部分的计算
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			
			if (extrinsic_est_en)		// 是否估计外参？这决定了是否对外参位置，即6和9号位的状态进行更新。
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{   
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(point_crossmat * C);
				// V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
				
				// h_x 是残差对所有状态量的偏导，也是线性化矩阵，对应 eq(13)
				// h_x，有m行，m即为当前这次迭代时，用到的lidar的点数。
				// 论文中 eq(13)，求偏导，对 位置（eq(13)的第一项）、旋转（eq(13)的第二项）有数值，其他的都是0；
				// 代码中，速度是3-5，旋转是0-2，即0-2是eq(13)的 u^T, 3-5是 -u^T R [p]，剩下的外参(6个)是后面的0.0，而没有出现省下的18维的0（速度，加速度，角速度，重力，ba，bg）
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
			// 计算残差，eq(12)，即公式的第一行。
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			m++;
		}
	}
	effct_feat_num += effect_num_k;
}


//~ The IMU observation model. eq(7)/(8), eq(14)
// IMU的观测模型，同时计算了残差，以及观测噪声R。对应论文eq(7)，eq(8), eq(14)，eq(16)
// 注意并没有计算H_I，和D_I，因为这些都是单位阵，直接在后面update时简单使用即可（update_iterated_dyn_share_IMU中进行的update）。
// 同时，判断了各通道是否饱和，如果饱和了，在update时就不利用IMU的数据进行更新了，因为已经测得不对了。此时只有在predict时能改变加饱和的状态。
void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
    std::memset(ekfom_data.satu_check, false, 6);
	ekfom_data.z_IMU.block<3,1>(0, 0) = angvel_avr - s.omg - s.bg;						// eq(14)，加速度的残差
	ekfom_data.z_IMU.block<3,1>(3, 0) = acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;		// eq(14)，角速度的残差
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;	// eq(16) 中的
	if(check_satu)
	{
		if(fabs(angvel_avr(0)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[0] = true; 
			ekfom_data.z_IMU(0) = 0.0;
		}
		
		if(fabs(angvel_avr(1)) >= 0.99 * satu_gyro) 
		{
			ekfom_data.satu_check[1] = true;
			ekfom_data.z_IMU(1) = 0.0;
		}
		
		if(fabs(angvel_avr(2)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[2] = true;
			ekfom_data.z_IMU(2) = 0.0;
		}
		
		if(fabs(acc_avr(0)) >= 0.99 * satu_acc)
		{
			ekfom_data.satu_check[3] = true;
			ekfom_data.z_IMU(3) = 0.0;
		}

		if(fabs(acc_avr(1)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[4] = true;
			ekfom_data.z_IMU(4) = 0.0;
		}

		if(fabs(acc_avr(2)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[5] = true;
			ekfom_data.z_IMU(5) = 0.0;
		}
	}
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{    
    V3D p_body(pi->x, pi->y, pi->z);
    
    V3D p_global;
	if (extrinsic_est_en)
	{	
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (kf_output.x_.offset_R_L_I * p_body + kf_output.x_.offset_T_L_I) + kf_output.x_.pos;
		}
		else
		{
			p_global = kf_input.x_.rot * (kf_input.x_.offset_R_L_I * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
		}
	}
	else
	{
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_output.x_.pos;
		}
		else
		{
			p_global = kf_input.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_input.x_.pos;
		}
	}

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};