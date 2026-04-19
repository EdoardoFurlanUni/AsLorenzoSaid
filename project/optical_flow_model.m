function y = optical_flow_model(x)
    % optical_flow_model - Optical Flow Measurement Model h(x)
    % Task 1
    % Implements the measurement function that maps the UAV state to the
    % quantities observed by the optical flow and distance sensors.
    %
    % Input:
    %   x : 16x1 state vector
    %       [q0, q1, q2, q3,      % quaternion attitude
    %        vn, ve, vd,           % velocity NED (m/s)
    %        pn, pe, pd,           % position NED (m)
    %        wb_x, wb_y, wb_z,    % gyro biases (rad/s)
    %        ab_x, ab_y, ab_z]    % accel biases (m/s^2)
    %
    % Output:
    %   y : 3x1 measurement vector
    %       [v_body_x;   % body-frame velocity X (m/s)  <- optical flow
    %        v_body_y;   % body-frame velocity Y (m/s)  <- optical flow
    %        p_d]        % position down / altitude (m) <- barometer

    q   = x(1:4);
    v_n = x(5:7);
    p_d = x(10);

    R_n2b = quat2rotm(q')';   % Rotation matrix: NED -> body

    v_b = R_n2b * v_n;        % Velocity rotated into body frame

    y = [v_b(1);
         v_b(2);
         p_d];
end
