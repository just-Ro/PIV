
function xyz_ret = PIV_pcfromdepth(imdepth, K)
% imdepth - depth image (in mm)
% K intrinsics

im_size=size(imdepth);
im_vec=imdepth(:);
Kx = K(1,1);
Cx = K(1,3);
Ky = K(2,2);
Cy = K(2,3);

u = repmat(1:im_size(2),im_size(1),1);
u = u(:)-Cx;
v = repmat((1:im_size(1))',im_size(2),1);
v=v(:)-Cy;
xyz=zeros(length(u),3);

xyz(:,3) = double(im_vec)*0.001; % Convert from mm to meters
xyz(:,1) = (xyz(:,3)/Kx) .* u ;
xyz(:,2) = (xyz(:,3)/Ky) .* v;

%plot3(x,y,z,'.');axis equal
xyz_ret = xyz;
end