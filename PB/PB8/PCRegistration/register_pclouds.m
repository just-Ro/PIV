% READ IMAGES and GENERATE POINT CLOUDS
load calib_asus.mat
im1=imread('rgb_image1_3.png');
im2=imread('rgb_image2_3.png');
load('depth1_3.mat')
dep1=depth_array;
load('depth2_3.mat')
dep2=depth_array;
figure(1);
imagesc([im1 im2]);
dep2(find(dep2(:)>4000))=0;

xyz1=get_xyzasus(dep1(:),[480 640],(1:640*480)', Depth_cam.K,1,0);
xyz2=get_xyzasus(dep2(:),[480 640],(1:640*480)', Depth_cam.K,1,0);
%REGISTER RGB TO DEPTH
rgbd1 = get_rgbd(xyz1, im1, R_d_to_rgb, T_d_to_rgb, RGB_cam.K);
rgbd2 = get_rgbd(xyz2, im2, R_d_to_rgb, T_d_to_rgb, RGB_cam.K);
figure(2)
imagesc([rgbd1 rgbd2]);
pc1=pointCloud(xyz1,'Color',reshape(rgbd1,[480*640 3]));
pc2=pointCloud(xyz2,'Color',reshape(rgbd2,[480*640 3]));
figure(3);
subplot(121);
showPointCloud(pc1);
view([0.52 -67.78])
subplot(122);
showPointCloud(pc2);
view([-91.36 -76.52])

%%
%GET CORRESPONDING POINTS IN THE 2 IMAGES (6 pts)
close all
np=6;
figure(1);imagesc(rgbd1 );
figure(2);imagesc(rgbd2 );
figure(1);x1=zeros(np,1);y1=x1;x2=y1;y2=x1;
for i=1:np,
    figure(1);
    [xa ya]=ginput(1);text(xa,ya,int2str(i));
    xa=fix(xa);ya=fix(ya);
    x1(i)=xa;y1(i)=ya;
    aux1=xyz1(sub2ind([480 640],ya,xa),:);
    figure(2);
    [xa ya]=ginput(1);text(xa,ya,int2str(i));
    xa=fix(xa);ya=fix(ya);
    x2(i)=xa;y2(i)=ya;
end
figure(3);
imshow([rgbd1 rgbd2]);
hold on; 
plot(x1,y1,'*r');
plot(x2+640,y2,'*r');
hold off;
ind1=sub2ind(size(dep2),y1,x1);
ind2=sub2ind(size(dep2),y2,x2);

P1=xyz1(ind1,:);
P2=xyz2(ind2,:);
%just in case clicked on invalid points
inds=find((P1(:,3).*P2(:,3))>0);
P1=P1(inds,:);P2=P2(inds,:);
%solve for rotation and translation
[d,xx,tr]=procrustes(P1,P2,'scaling',false,'reflection',false);
%Transform pointcloud 2 to the world (camera 1).
xyz21=xyz2*tr.T+ones(length(xyz2),1)*tr.c(1,:);
pc1=pointCloud(xyz1,'Color',reshape(rgbd1,[480*640 3]));
pc2=pointCloud(xyz21,'Color',reshape(rgbd2,[480*640 3]));
figure(4);
showPointCloud(pc1);
view([-2.89 -74.53])
figure(5);
showPointCloud(pc2);
view([0.22 -79.91])

%% 
%SHOW ALL CLOUDS FUSING
figure(6)
pcshow(pcmerge(pc1,pc2,0.001));    
view([-0.60 -74.06])

