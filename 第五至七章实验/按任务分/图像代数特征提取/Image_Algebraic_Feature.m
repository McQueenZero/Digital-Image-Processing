%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ͼ�����������ȡ
% ע�⣺   ��С������
%%-------------------------------------------------------------------------
% Reference: https://microstrong.blog.csdn.net/article/details/80632779
% PCA�����ɷַ�������������ֵ�ֽ�Э�������
% SVD������ֵ�ֽ⣬����SVD�ֽ�Э�������
% Reference: https://blog.csdn.net/qq_37766828/article/details/89407596
% KPCA: �����ɷַ����������ں˺���
%%-------------------------------------------------------------------------
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