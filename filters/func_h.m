function [z] = func_h(sym_x)
    %% h(x) - Optical Flow Measurement Model
    % sym_x: 16x1 symbolic state vector
    % 1:4   - Quaternions (q0, q1, q2, q3)
    % 5:7   - Velocity NED (vn, ve, vd)
    % 8:10  - Position NED (pn, pe, pd)
    
    q = sym_x(1:4);
    v_n = sym_x(5:7);
    p_d = sym_x(10);
    
    % Direction Cosine Matrix (Body to Nav) - R_b2n
    % We need Nav to Body (R_n2b), which is the transpose
    q0 = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
    
    R_b2n = [q0^2 + q1^2 - q2^2 - q3^2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2); ...
             2*(q1*q2 + q0*q3), q0^2 - q1^2 + q2^2 - q3^2, 2*(q2*q3 - q0*q1); ...
             2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0^2 - q1^2 - q2^2 + q3^2];
    
    R_n2b = R_b2n';
    
    v_b = R_n2b * v_n;
    
    % Measurements: [v_body_x; v_body_y; pos_ned_z]
    z = [v_b(1);
         v_b(2);
         p_d];
end

