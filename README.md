# AsLorenzoSaid - Learning Dynamical Systems

This project focuses on **Learning Dynamical Systems**, specifically for **Optical Flow Estimation** using filtered data (EKF, UKF). The implementation is based on the [PX4-Autopilot v1.16.0](https://github.com/PX4/PX4-Autopilot).

## Project Structure

- **`Data/`**: Contains optical flow datasets. Note that raw and processed data are excluded from version control via `.gitignore`.
  - `estimator_optical_flow`: Velocity in NED frame using EKF.
  - `vehicle_optical_flow`: Processed optical flow data (aligned using sampling time).
  - `sensor_optical_flow`: Raw sensor data.
- **`filters/`**: Core implementations of various Kalman Filters:
  - `EKF.m` / `REKF.m`: Extended Kalman Filter / Robust EKF.
  - `UKF.m` / `RUKF.m`: Unscented Kalman Filter / Robust UKF.
  - `func_f.m`, `func_h.m`: System dynamics and measurement functions.
- **`project/`**: Contains specific analysis and verification tasks.
- **`presentation/`**: Presentation materials for the course.

## Project Tasks (Detailed Breakdown)

The project is organized into several sequential tasks located in the `project/` directory. Each task builds up the physical implementation of a **GPS-Denied** UAV filtering structure using Optical Flow and Barometer data.

### **Task 1: Optical Flow Model (`optical_flow_model.m`)**
- **Objective:** Definition of the physical State-Space model for the Optical Flow tracking.
- **Details:** Maps how the unrotated NED (North-East-Down) Earth velocities translate into the local `Body X` and `Body Y` speeds measured by the bottom-facing camera, applying the necessary Rotational Matrices obtained from Quaternions.

### **Task 2: Model Accuracy Verification (`task2.m`)**
- **Objective:** Validating that our physical model ($h(x)$) accurately translates the real world without errors.
- **Details:** Evaluates the `optical_flow_model` by dynamically feeding it the exact Ground Truth GPS Velocities and the PX4 estimated Quaternions (`q_sync`). The resulting predicted `v_body` velocities seamlessly match the real raw sensor flows from the dataset, verifying that the algebraic matrices inside `func_h` and `func_f` are mathematically sound.

### **Task 3: EKF & UKF Implementation (`task3.m`)**
- **Objective:** State approximation through Extended and Unscented Kalman Filtering.
- **Engineering Changes:**
  - **Rejection of Symbolic Calculation:** We bypassed the original pre-packaged `EKF.m` template because its internal usage of `subs()` (Symbolic Substitution) would require roughly 10 hours for 400 iterations. We generated ultra-fast binary compiled numerical blocks using `matlabFunction()`, retaining execution times within fractions of a second.
  - **IMU Handling:** We specifically injected the raw gyro and accel matrices (`U_i` Control Inputs) within the prediction loop to follow $x_{k+1} = f(x_k, u_k, dt)$.
  - **Filter Tuning:** In GPS-Denied modes, heading (Yaw) becomes strictly unobservable via generic models, generating expected but contained "drift". We compensated for Barometer discretization noise increasing standard deviations to enforce strong covariance checks. Forced quaternion normalization is applied to each closure loop to circumvent norm divergence.

### **Task 4: Robust Filtering - REKF & RUKF (`task4.m`)**
- **Objective:** Implement **Robust** filtering variations against sensor outliers (glitches, spikes in distances, or lens scale mismatches).
- **Engineering Changes:**
  - Standard files (`REKF.m` and `RUKF.m` found in `filters/`) were not natively usable for dynamical systems taking time-varying inputs.
  - We generated two separate robust scripts: **`REKF_fast.m`** and **`RUKF_fast.m`**, optimizing them.
  - **Matrix Fault Protection:** Calculating least-favorable bounding (Robust estimation by multiplying matrices natively shrinks them boundedly) exposed mathematical vulnerabilities to $10^{-16}$ float discrepancies causing `Matrix must be positive definite` closures (`chol()` explosions). We protected the filters by implementing strict SVD bounding constraints (`svd()`) and direct algebraic inverse avoidance ($V * (I - \theta V)^{-1}$), making the robust logic practically indestructible to numerical spikes.

## 🎯 TODOs for Final Verification & Analysis

The following components still require formal verification and analysis before the final report is compiled:

- [ ] **Data Preprocessing (IMU Scale Bug):** Verify the recent correction made to `DATA_PROCESS.m` and `sync_all_sensors.m`. A critical bug was fixed where the IMU increments (`delta_velocity` and `delta_angle` integrated at ~227Hz) were being incorrectly passed to the 100Hz filter step without first being converted to true rates, causing gravity to accumulate un-canceled and the filter to drift out of scale. Ensure the fix properly holds up across all datasets.
- [ ] **Filter Smoothing & Tuning:** Despite fixing the scaling, the outputs of the filters track high-frequency propeller vibrations originating from the raw IMU. Test the application of a moving average filter (e.g., `movmean(acc, 15)`) directly in `sync_all_sensors.m` and lower the Velocity Process Covariance in `B_mat` to heavily smooth out the final EKF/UKF tracking.
- [ ] **Task 1 (Model Definition):** Verify and justify the mathematical formulation of the IMU state dynamics (`func_f.m`). Specifically, analyze why the equations use discrete increments instead of the continuous-time rates proposed in the course presentation.
- [ ] **Tasks 3 & 4 (Filter Performance & Drift):** Conduct an in-depth analysis of the resulting velocities compared to the GPS Ground Truth. Investigate the causes behind the visible drifting behavior and justify whether this is a filter anomaly or a physical consequence of the sensors used.
- [ ] **Task 3 & 4 (EKF vs UKF Comparison):** Provide a structured comparison between the results of the Extended and Unscented Kalman Filters to determine which is more robust for our specific non-linear setup.

## Data Handling

To avoid uploading large datasets and logs:
- `.gitignore` is used to exclude `.mat` files and `log_*/` directories within `Data/mat/`.
- Refer to `Data/README.txt` for more details on the data structure.

## Requirements
- MATLAB
- (Optional) Navigation Toolbox for conversions (e.g., `quat2rotm`).