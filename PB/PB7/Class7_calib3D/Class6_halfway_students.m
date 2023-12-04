d=dir('Calibration*.mat');
load calib_asus.mat
%%
%read all data to a single structure;
dados={};
for i=1:length(d),
    load(d(i).name);
    calib_data.xyz=xyz;
    dados{i}=calib_data;
    figure(1);imagesc(calib_data.image);
    im=get_rgbd(xyz,calib_data.image,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);
    figure(2);imagesc(im);
    figure(3);showPointCloud(pointCloud(xyz,'Color',reshape(im,[640*480 3])));
    i,
    pause(1);
end

%%
%Pick 2 images and relate them 
%RGB image
im1=dados{1}.image;
im2=dados{15}.image;
%RGB image registered in depth
imd1=get_rgbd(dados{1}.xyz,im1,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);
imd2=get_rgbd(dados{15}.xyz,im2,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);
%show them side by side
figure(1);
imagesc([imd1 im1]);
figure(2);
imagesc([imd2 im2]);
pc1=pointCloud(dados{1}.xyz);
pc2=pointCloud(dados{15}.xyz,'Color',reshape(imd2,[640*480 3]));
%Point cloud second image
figure(3);showPointCloud(pc2);

%%
% Write a script that accepts a coordinate in figure(2) - second camera and
% predicts where that point is in image 1 !