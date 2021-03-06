%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像增强
%       直方图有关操作、空域滤波、图像代数特征提取等
% 注意：   分小节运行
%%-------------------------------------------------------------------------
%% 灰度图像直方图
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = imSrc;
imGrayhisteq = histeq(imGray,256);   %直方图均衡化
figure('Name','直方图','NumberTitle','off')
subplot(1,2,1),imhist(imGray),title('直方图');    %直方图
subplot(1,2,2),imhist(imGrayhisteq),title('均衡化处理过的直方图');
figure('Name','灰度图像','NumberTitle','off')
subplot(1,2,1),imshow(imGray),title('灰度图片');
subplot(1,2,2),imshow(imGrayhisteq),title('均衡化处理过的灰度图');

%% 彩色图像直方图
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imRGBhisteq = histeq(imSrc,256);   %直方图均衡化
figure('Name','直方图','NumberTitle','off')
subplot(1,2,1),imhist(imSrc),title('直方图');    %直方图
subplot(1,2,2),imhist(imRGBhisteq),title('均衡化处理过的直方图');
figure('Name','彩色图像','NumberTitle','off')
subplot(1,2,1),imshow(imSrc),title('彩色图片');
subplot(1,2,2),imshow(imRGBhisteq),title('均衡化处理过的彩色图');

%% 空域滤波图像平滑
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imNosSP=imnoise(imSrc,'salt & pepper',0.02);
imNosGAS=imnoise(imSrc,'gaussian');
figure('Name','原始图像及噪声图像','NumberTitle','off')
subplot(1,3,1),imshow(imSrc),title('原始图片');
subplot(1,3,2),imshow(imNosSP),title('椒盐噪声图片');
subplot(1,3,3),imshow(imNosGAS),title('高斯噪声图片');

imDstSP3x3=medfilt2(imNosSP,[3 3]);
imDstSP5x5=medfilt2(imNosSP,[5 5]);
imDstGAS3x3=medfilt2(imNosSP,[3 3]);
imDstGAS5x5=medfilt2(imNosSP,[5 5]);
figure('Name','中值滤波去噪对比','NumberTitle','off')
subplot(2,2,1),imshow(imDstSP3x3),title('空域3x3中值滤波去椒盐噪声图片');
subplot(2,2,2),imshow(imDstSP5x5),title('空域5x5中值滤波去椒盐噪声图片');
subplot(2,2,3),imshow(imDstGAS3x3),title('空域3x3中值滤波去高斯噪声图片');
subplot(2,2,4),imshow(imDstGAS5x5),title('空域5x5中值滤波去高斯噪声图片');

%% 拉普拉斯算子图像锐化
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
m4 = fspecial('laplacian',0);   %中心为-4的拉普拉斯滤波器
m8 = [1 1 1; 1 -8 1; 1 1 1];    %中心为-8的拉普拉斯滤波器
imf = im2double(imSrc);
imDstm4 = imf-imfilter(imf,m4,'symmetric');
imDstm8 = imf-imfilter(imf,m8,'symmetric');
figure('Name','图像锐化','NumberTitle','off')
subplot(1,3,1),imshow(imSrc),title('原始图片');
subplot(1,3,2),imshow(imDstm4),title('中心为-4的拉普拉斯滤波器处理所得图片');
subplot(1,3,3),imshow(imDstm8),title('中心为-8的拉普拉斯滤波器处理所得图片');

%% 图像代数特征提取（PCA方法）
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = rgb2gray(imSrc);   %RGB转灰度图
imTemp = im2double(imGray);
% 主成分分析法
% COEFF：是主成分分量，即样本协方差矩阵的特征向量
% SCORE：主成分，是样本X在低维空间的表示形式，
% 即样本X在主成份分量COEFF上的投影 ，若需要降k维，则只需要取前k列主成分分量即可
% latent：一个包含样本协方差矩阵特征值的向量
% 每行是一个样本，每列是一个特征，(2,4)表示第二个样本的第四个特征
[COEFF,SCORE,latent]=pca(imTemp);
k = 3;
figure(1),plot(SCORE(:,k))    %第k个特征
xlabel('样本序号'),ylabel('特征值')
title(['第' num2str(k) '个特征'])
axis tight, grid on
m = 5;
figure(2),plot(SCORE(m,:))    %第m个样本
xlabel('特征序号'),ylabel('特征值')
title(['第' num2str(m) '个样本'])
axis tight, grid on
% 主特征：特征序号靠前（如第1~第50个特征）的特征值
% 降维：提取前n个主成分
n = 3;
pcaData = SCORE(:,1:n);
figure(3),plot(pcaData)
xlabel('样本序号'),ylabel('特征值')
title(['前' num2str(n) '个主成分'])
axis tight, grid on
figure(4),scatter3(pcaData(:,1),pcaData(:,2),pcaData(:,3),'filled')


%% 图像代数特征提取（SVD方法）
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = rgb2gray(imSrc);   %RGB转灰度图
imTemp = im2double(imGray);
% 奇异值分解法
% U：左奇异矩阵
% S：对角矩阵，对角线上的元素是奇异值，从大到小排列
% V：右奇异矩阵
[U,S,V] = svd(imTemp);
% 降维：提取前n个非零奇异值
n = 3;
U_1 = U(:,1:n);
S_1 = S(:,1:n);
V_1 = V(:,1:n);
svdData = U_1*S_1'*V_1;
figure(1),plot(svdData)
xlabel('样本序号'),ylabel('奇异值')
title(['前' num2str(n) '个非零奇异值'])
axis tight, grid on

%% 图像代数特征提取（KPCA方法）
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = rgb2gray(imSrc);   %RGB转灰度图
imTemp = im2double(imGray);

data=imTemp;
[Xrow, Xcol] = size(data);    % Xrow：样本个数 Xcol：样本属性个数
%%数据预处理
Xmean = mean(data); % 求原始数据的均值
Xstd = std(data); % 求原始数据的方差
X0 = (data-ones(Xrow,1)*Xmean) ./ (ones(Xrow,1)*Xstd); % 标准阵X0,标准化为均值0，方差1;
c = 20000; %此参数可调
%%求核矩阵
for i = 1 : Xrow
for j = 1 : Xrow
%k(i,j)=kernel(data(i,:),data(j,:),2,6);   
K(i,j) = exp(-(norm(X0(i,:) - X0(j,:)))^2/c);%求核矩阵，采用径向基核函数，参数c
end
end
%%中心化矩阵
unit = (1/Xrow) * ones(Xrow, Xrow);
Kp = K - unit*K - K*unit + unit*K*unit; % 中心化矩阵
%%特征值分解
[eigenvector, eigenvalue] = eig(Kp); % 求协方差矩阵的特征向量（eigenvector）和特征值（eigenvalue）
%单位化特征向量
for m =1 : hei
for n =1 : wid
Normvector(n,m) = eigenvector(n,m)/sum(eigenvector(:,m));
end
end
eigenvalue_vec = real(diag(eigenvalue)); %将特征值矩阵转换为向量
[eigenvalue_sort, index] = sort(eigenvalue_vec, 'descend'); % 特征值按降序排列，eigenvalue_sort是排列后的数组，index是序号
pcIndex = []; % 记录主元所在特征值向量中的序号
pcn = 3;
for k = 1 : pcn % 特征值个数
pcIndex(k) = index(k); % 保存主元序号
end
for i = 1 : pcn
pc_vector(i) = eigenvalue_vec(pcIndex(i)); % 主元向量
P(:, i) = Normvector(:, pcIndex(i)); % 主元所对应的特征向量（负荷向量）
end
project_invectors = k*P;
pc_vector2 = diag(pc_vector); % 构建主元对角阵 
% 降维：提取前3个主元
% 绘制三维散点图
x=project_invectors(:,1);
y=project_invectors(:,2);
z=project_invectors(:,3);
scatter3(x,y,z,'filled')
