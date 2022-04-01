%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ͼ����ǿ
%       ֱ��ͼ�йز����������˲���ͼ�����������ȡ��
% ע�⣺   ��С������
%%-------------------------------------------------------------------------
%% �Ҷ�ͼ��ֱ��ͼ
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = imSrc;
imGrayhisteq = histeq(imGray,256);   %ֱ��ͼ���⻯
figure('Name','ֱ��ͼ','NumberTitle','off')
subplot(1,2,1),imhist(imGray),title('ֱ��ͼ');    %ֱ��ͼ
subplot(1,2,2),imhist(imGrayhisteq),title('���⻯�������ֱ��ͼ');
figure('Name','�Ҷ�ͼ��','NumberTitle','off')
subplot(1,2,1),imshow(imGray),title('�Ҷ�ͼƬ');
subplot(1,2,2),imshow(imGrayhisteq),title('���⻯������ĻҶ�ͼ');

%% ��ɫͼ��ֱ��ͼ
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imRGBhisteq = histeq(imSrc,256);   %ֱ��ͼ���⻯
figure('Name','ֱ��ͼ','NumberTitle','off')
subplot(1,2,1),imhist(imSrc),title('ֱ��ͼ');    %ֱ��ͼ
subplot(1,2,2),imhist(imRGBhisteq),title('���⻯�������ֱ��ͼ');
figure('Name','��ɫͼ��','NumberTitle','off')
subplot(1,2,1),imshow(imSrc),title('��ɫͼƬ');
subplot(1,2,2),imshow(imRGBhisteq),title('���⻯������Ĳ�ɫͼ');

%% �����˲�ͼ��ƽ��
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imNosSP=imnoise(imSrc,'salt & pepper',0.02);
imNosGAS=imnoise(imSrc,'gaussian');
figure('Name','ԭʼͼ������ͼ��','NumberTitle','off')
subplot(1,3,1),imshow(imSrc),title('ԭʼͼƬ');
subplot(1,3,2),imshow(imNosSP),title('��������ͼƬ');
subplot(1,3,3),imshow(imNosGAS),title('��˹����ͼƬ');

imDstSP3x3=medfilt2(imNosSP,[3 3]);
imDstSP5x5=medfilt2(imNosSP,[5 5]);
imDstGAS3x3=medfilt2(imNosSP,[3 3]);
imDstGAS5x5=medfilt2(imNosSP,[5 5]);
figure('Name','��ֵ�˲�ȥ��Ա�','NumberTitle','off')
subplot(2,2,1),imshow(imDstSP3x3),title('����3x3��ֵ�˲�ȥ��������ͼƬ');
subplot(2,2,2),imshow(imDstSP5x5),title('����5x5��ֵ�˲�ȥ��������ͼƬ');
subplot(2,2,3),imshow(imDstGAS3x3),title('����3x3��ֵ�˲�ȥ��˹����ͼƬ');
subplot(2,2,4),imshow(imDstGAS5x5),title('����5x5��ֵ�˲�ȥ��˹����ͼƬ');

%% ������˹����ͼ����
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
m4 = fspecial('laplacian',0);   %����Ϊ-4��������˹�˲���
m8 = [1 1 1; 1 -8 1; 1 1 1];    %����Ϊ-8��������˹�˲���
imf = im2double(imSrc);
imDstm4 = imf-imfilter(imf,m4,'symmetric');
imDstm8 = imf-imfilter(imf,m8,'symmetric');
figure('Name','ͼ����','NumberTitle','off')
subplot(1,3,1),imshow(imSrc),title('ԭʼͼƬ');
subplot(1,3,2),imshow(imDstm4),title('����Ϊ-4��������˹�˲�����������ͼƬ');
subplot(1,3,3),imshow(imDstm8),title('����Ϊ-8��������˹�˲�����������ͼƬ');

%% ͼ�����������ȡ��PCA������
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = rgb2gray(imSrc);   %RGBת�Ҷ�ͼ
imTemp = im2double(imGray);
% ���ɷַ�����
% COEFF�������ɷַ�����������Э����������������
% SCORE�����ɷ֣�������X�ڵ�ά�ռ�ı�ʾ��ʽ��
% ������X�����ɷݷ���COEFF�ϵ�ͶӰ ������Ҫ��kά����ֻ��Ҫȡǰk�����ɷַ�������
% latent��һ����������Э�����������ֵ������
% ÿ����һ��������ÿ����һ��������(2,4)��ʾ�ڶ��������ĵ��ĸ�����
[COEFF,SCORE,latent]=pca(imTemp);
k = 3;
figure(1),plot(SCORE(:,k))    %��k������
xlabel('�������'),ylabel('����ֵ')
title(['��' num2str(k) '������'])
axis tight, grid on
m = 5;
figure(2),plot(SCORE(m,:))    %��m������
xlabel('�������'),ylabel('����ֵ')
title(['��' num2str(m) '������'])
axis tight, grid on
% ��������������ſ�ǰ�����1~��50��������������ֵ
% ��ά����ȡǰn�����ɷ�
n = 3;
pcaData = SCORE(:,1:n);
figure(3),plot(pcaData)
xlabel('�������'),ylabel('����ֵ')
title(['ǰ' num2str(n) '�����ɷ�'])
axis tight, grid on
figure(4),scatter3(pcaData(:,1),pcaData(:,2),pcaData(:,3),'filled')


%% ͼ�����������ȡ��SVD������
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = rgb2gray(imSrc);   %RGBת�Ҷ�ͼ
imTemp = im2double(imGray);
% ����ֵ�ֽⷨ
% U�����������
% S���ԽǾ��󣬶Խ����ϵ�Ԫ��������ֵ���Ӵ�С����
% V�����������
[U,S,V] = svd(imTemp);
% ��ά����ȡǰn����������ֵ
n = 3;
U_1 = U(:,1:n);
S_1 = S(:,1:n);
V_1 = V(:,1:n);
svdData = U_1*S_1'*V_1;
figure(1),plot(svdData)
xlabel('�������'),ylabel('����ֵ')
title(['ǰ' num2str(n) '����������ֵ'])
axis tight, grid on

%% ͼ�����������ȡ��KPCA������
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = rgb2gray(imSrc);   %RGBת�Ҷ�ͼ
imTemp = im2double(imGray);

data=imTemp;
[Xrow, Xcol] = size(data);    % Xrow���������� Xcol���������Ը���
%%����Ԥ����
Xmean = mean(data); % ��ԭʼ���ݵľ�ֵ
Xstd = std(data); % ��ԭʼ���ݵķ���
X0 = (data-ones(Xrow,1)*Xmean) ./ (ones(Xrow,1)*Xstd); % ��׼��X0,��׼��Ϊ��ֵ0������1;
c = 20000; %�˲����ɵ�
%%��˾���
for i = 1 : Xrow
for j = 1 : Xrow
%k(i,j)=kernel(data(i,:),data(j,:),2,6);   
K(i,j) = exp(-(norm(X0(i,:) - X0(j,:)))^2/c);%��˾��󣬲��þ�����˺���������c
end
end
%%���Ļ�����
unit = (1/Xrow) * ones(Xrow, Xrow);
Kp = K - unit*K - K*unit + unit*K*unit; % ���Ļ�����
%%����ֵ�ֽ�
[eigenvector, eigenvalue] = eig(Kp); % ��Э������������������eigenvector��������ֵ��eigenvalue��
%��λ����������
for m =1 : hei
for n =1 : wid
Normvector(n,m) = eigenvector(n,m)/sum(eigenvector(:,m));
end
end
eigenvalue_vec = real(diag(eigenvalue)); %������ֵ����ת��Ϊ����
[eigenvalue_sort, index] = sort(eigenvalue_vec, 'descend'); % ����ֵ���������У�eigenvalue_sort�����к�����飬index�����
pcIndex = []; % ��¼��Ԫ��������ֵ�����е����
pcn = 3;
for k = 1 : pcn % ����ֵ����
pcIndex(k) = index(k); % ������Ԫ���
end
for i = 1 : pcn
pc_vector(i) = eigenvalue_vec(pcIndex(i)); % ��Ԫ����
P(:, i) = Normvector(:, pcIndex(i)); % ��Ԫ����Ӧ����������������������
end
project_invectors = k*P;
pc_vector2 = diag(pc_vector); % ������Ԫ�Խ��� 
% ��ά����ȡǰ3����Ԫ
% ������άɢ��ͼ
x=project_invectors(:,1);
y=project_invectors(:,2);
z=project_invectors(:,3);
scatter3(x,y,z,'filled')
