%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��3��
% ˵����   Image Demosaic
%       ����һ��RGBͼ���Ƚ���RGB2Bayerת�����ٽ���Bayer2RGBת����
%       �ñ�Ե����Ӧ��ֵ�ķ�������ȥ�����˴���
%%-------------------------------------------------------------------------
%% ѡ��ȥ�����˷���
clc,clear,close all;
while 1
    disp("MATLAB�Դ�ȥ�����˺���������'1'")
    disp("3x3���ڲ�ֵ������'2'")
    disp("��Ե����Ӧ��ֵ������'3'")
    Method = input('����ȥ�����˷�����');
    switch Method
        case 1
            break
        case 2
            break
        case 3
            break
        otherwise
            disp('�Ƿ�������������')
    end
end

%% ��ȡԭʼͼ��
tic;
% �����ɫͼ
filename = 'Tokyo University Yasuda Koudou';
imSrc = imread([filename,'.jpg']);
figure('Name','ԭʼͼƬ','NumberTitle','off'),imshow(imSrc);
% imwrite(imSrc,'ԭʼ���ɫͼƬ.png')
[hei, wid, len] = size(imSrc);
bayer = uint8(zeros(hei,wid));
for k=1:3
    bayer_mosaic(:,:,k) = uint8(zeros(hei,wid));
end

%% RGBͼ��תRGGBģʽBayer������ͼ��
% RGGBģʽ����
% R G R G R G
% G B G B G B
% R G R G R G
% G B G B G B
% -----------
for ver = 1:hei
    for hor = 1:wid
        if((1 == mod(ver,2)) && (1 == mod(hor,2)))
            bayer(ver,hor) = imSrc(ver,hor,1);  %R
            bayer_mosaic(ver,hor,1) = imSrc(ver,hor,1);
        elseif((0 == mod(ver,2)) && (0 == mod(hor,2)))
            bayer(ver,hor) = imSrc(ver,hor,3);  %B
            bayer_mosaic(ver,hor,3) = imSrc(ver,hor,3);
        else
            bayer(ver,hor) = imSrc(ver,hor,2);  %G
            bayer_mosaic(ver,hor,2) = imSrc(ver,hor,2);
        end
    end
end

% figure('Name','RGGB�Ҷ�ͼƬ','NumberTitle','off'),imshow(bayer);
figure('Name','RGGB������ͼƬ','NumberTitle','off'),imshow(bayer_mosaic);
% imwrite(bayer_mosaic,'RGGB��Bayer��ʽ������ͼƬ.png')

%% MATLAB�Դ�ȥ�����˺��������Բ�ֵ��
if Method == 1
    imDst = demosaic(bayer,'rggb');
    figure('Name','MATLAB�Դ�ȥ�����˺�������ͼƬ','NumberTitle','off'),imshow(imDst);
    % imwrite(imDst,'MATLAB�Դ�ȥ�����˺�������ͼƬ.png');
    toc;
end
%% 3x3���ڲ�ֵȥ������
if Method == 2
    bayerPadding = zeros(hei+2,wid+2);
    bayerPadding(2:hei+1,2:wid+1) = bayer;
    bayerPadding(1,:) = bayerPadding(3,:);
    bayerPadding(hei+2,:) = bayerPadding(hei,:);
    bayerPadding(:,1) = bayerPadding(:,3);
    bayerPadding(:,wid+2) = bayerPadding(:,wid);
    imDst = zeros(hei+2, wid+2, len);

    for ver = 2:hei+1
        for hor = 2:wid+1
            if(1 == mod(ver-1,2))   %����R
                if(1 == mod(hor-1,2))   %����R
                    imDst(ver,hor,1) = bayerPadding(ver,hor);
                    imDst(ver,hor,3) = (bayerPadding(ver-1,hor-1) + bayerPadding(ver-1,hor+1) + bayerPadding(ver+1,hor-1) + bayerPadding(ver+1,hor+1)) / 4;
                    imDst(ver,hor,2) = (bayerPadding(ver-1,hor) + bayerPadding(ver,hor-1) + bayerPadding(ver,hor+1) + bayerPadding(ver+1,hor)) / 4;
                else                    %����G
                    imDst(ver,hor,2) = bayerPadding(ver,hor);
                    imDst(ver,hor,3) = (bayerPadding(ver-1,hor) + bayerPadding(ver+1,hor)) / 2;
                    imDst(ver,hor,1) = (bayerPadding(ver,hor-1) + bayerPadding(ver,hor+1)) / 2;
                end
            else                    %����G
                if(1 == mod(hor-1,2))   %����G
                    imDst(ver,hor,2) = bayerPadding(ver,hor);
                    imDst(ver,hor,3) = (bayerPadding(ver,hor-1) + bayerPadding(ver,hor+1)) / 2;
                    imDst(ver,hor,1) = (bayerPadding(ver-1,hor) + bayerPadding(ver+1,hor)) / 2;
                else                    %����B
                    imDst(ver,hor,3) = bayerPadding(ver,hor);
                    imDst(ver,hor,2) = (bayerPadding(ver-1,hor) + bayerPadding(ver,hor-1) + bayerPadding(ver,hor+1) + bayerPadding(ver+1,hor)) / 4;
                    imDst(ver,hor,1) = (bayerPadding(ver-1,hor-1) + bayerPadding(ver-1,hor+1) + bayerPadding(ver+1,hor-1) + bayerPadding(ver+1,hor+1)) / 4;
                end
            end
        end
    end

    imDst = uint8(imDst(2:hei+1,2:wid+1,:));
    figure('Name','3x3���ڲ�ֵȥ������ͼƬ','NumberTitle','off'),imshow(imDst);
    % imwrite(imDst,'3x3���ڲ�ֵȥ������ͼƬ.png');
    toc;
end
%% ��Ե����Ӧ��ֵȥ������
% RGGBģʽ����
% R G R G R G
% G B G B G B
% R G R G R G
% G B G B G B
% ����ģ������
% B G B   R G R   G R G   G B G
% G R G   G B G   B G B   R G R
% B G B   R G R   G R G   G B G
%  (a)     (b)     (c)     (d)
% -----------
if Method == 3
    bayerPadding = zeros(hei+4,wid+4);
    bayerPadding(3:hei+2,3:wid+2) = bayer;
    bayerPadding(1,:) = bayerPadding(5,:);
    bayerPadding(2,:) = bayerPadding(4,:);
    bayerPadding(hei+4,:) = bayerPadding(hei,:);
    bayerPadding(hei+3,:) = bayerPadding(hei+1,:);
    bayerPadding(:,1) = bayerPadding(:,5);
    bayerPadding(:,2) = bayerPadding(:,4);
    bayerPadding(:,wid+4) = bayerPadding(:,wid);
    bayerPadding(:,wid+3) = bayerPadding(:,wid+1);
    imDst = zeros(hei+4, wid+4, len);

    for ver = 3:hei+2   %�±�i���к�
        for hor = 3:wid+2   %�±�j���к�
            if mod(ver-2,2)==1  %������
                if mod(hor-2,2)==1  %�����У�Rλ���ж�
                    imDst(ver,hor,1) = bayerPadding(ver,hor);
                    if delta_H(ver,hor,bayerPadding)<delta_V(ver,hor,bayerPadding)
                        imDst(ver,hor,2) = GH(ver,hor,bayerPadding);
                    elseif delta_H(ver,hor,bayerPadding)>delta_V(ver,hor,bayerPadding)
                        imDst(ver,hor,2) = GV(ver,hor,bayerPadding);
                    else
                        imDst(ver,hor,2) = GM(ver,hor,bayerPadding);
                    end
                else                %ż���У�Gλ���ж�
                    imDst(ver,hor,2) = bayerPadding(ver,hor);
                end
            else                %ż����
                if mod(hor-2,2)==0  %ż���У�Bλ���ж�
                    imDst(ver,hor,3) = bayerPadding(ver,hor);
                    if delta_H(ver,hor,bayerPadding)<delta_V(ver,hor,bayerPadding)
                        imDst(ver,hor,2) = GH(ver,hor,bayerPadding);
                    elseif delta_H(ver,hor,bayerPadding)>delta_V(ver,hor,bayerPadding)
                        imDst(ver,hor,2) = GV(ver,hor,bayerPadding);
                    else
                        imDst(ver,hor,2) = GM(ver,hor,bayerPadding);
                    end
                else                %�����У�Gλ���ж�
                    imDst(ver,hor,2) = bayerPadding(ver,hor);
                end
            end
        end
    end

    for ver = 3:hei+2   %�±�i���к�
        for hor = 3:wid+2   %�±�j���к�
            if mod(ver-2,2)==1  %������
                if mod(hor-2,2)==1  %�����У�Rλ���ж�
                    if D_45(ver,hor,bayerPadding,imDst)<D_135(ver,hor,bayerPadding,imDst)
                        imDst(ver,hor,3) = RB45(ver,hor,bayerPadding,imDst);
                    elseif D_45(ver,hor,bayerPadding,imDst)>D_135(ver,hor,bayerPadding,imDst)
                        imDst(ver,hor,3) = RB135(ver,hor,bayerPadding,imDst);
                    else
                        imDst(ver,hor,3) = RBM(ver,hor,bayerPadding,imDst);
                    end
                else
                    imDst(ver,hor,3) = RBV(ver,hor,bayerPadding,imDst);
                    imDst(ver,hor,1) = RBH(ver,hor,bayerPadding,imDst);
                end
            else                %ż����
                if mod(hor-2,2)==0  %ż���У�Bλ���ж�
                    if D_45(ver,hor,bayerPadding,imDst)<D_135(ver,hor,bayerPadding,imDst)
                        imDst(ver,hor,1) = RB45(ver,hor,bayerPadding,imDst);
                    elseif D_45(ver,hor,bayerPadding,imDst)>D_135(ver,hor,bayerPadding,imDst)
                        imDst(ver,hor,1) = RB135(ver,hor,bayerPadding,imDst);
                    else
                        imDst(ver,hor,1) = RBM(ver,hor,bayerPadding,imDst);
                    end
                else
                    imDst(ver,hor,1) = RBV(ver,hor,bayerPadding,imDst);
                    imDst(ver,hor,3) = RBH(ver,hor,bayerPadding,imDst);
                end
            end
        end
    end

    imDst = uint8(imDst(3:hei+2,3:wid+2,:));
    figure('Name','��Ե����Ӧ��ֵȥ������ͼƬ','NumberTitle','off'),imshow(imDst);
    % imwrite(imDst,'��Ե����Ӧ��ֵȥ������ͼƬ.png');
    toc;
end
%% ��Ե����Ӧ��ֵ��Ƕ����
% ��ɫ�����ؽ�
function HO=delta_H(ii,jj,bayer)  %ˮƽ�������
HO=abs(bayer(ii,jj-1)-bayer(ii,jj+1))+ ...
    abs(2*bayer(ii,jj)-bayer(ii,jj-2)-bayer(ii,jj+2));
end

function VO=delta_V(ii,jj,bayer)  %��ֱ�������
VO=abs(bayer(ii-1,jj)-bayer(ii+1,jj))+ ...
    abs(2*bayer(ii,jj)-bayer(ii-2,jj)-bayer(ii+2,jj));
end

function g=GH(ii,jj,bayer)        %��ɫ������ˮƽ�������
g=(bayer(ii,jj-1)+bayer(ii,jj+1))/2+ ...
    (2*bayer(ii,jj)-bayer(ii,jj-2)-bayer(ii,jj+2))/4;
end

function g=GV(ii,jj,bayer)        %��ɫ�����ش�ֱ�������
g=(bayer(ii-1,jj)+bayer(ii+1,jj))/2+ ...
    (2*bayer(ii,jj)-bayer(ii-2,jj)-bayer(ii+2,jj))/4;
end

function g=GM(ii,jj,bayer)        %��ɫ����Ϊ����������ƽ��ֵ
g=(GH(ii,jj,bayer)+GV(ii,jj,bayer))/2;
end

% ��ɫ�����㴦�ĺ�ɫ����ɫ�����ؽ�
function rb=RBV(ii,jj,bayer,rgb)      %��/��ɫ����ʹ����������R/B-G�ռ�����Բ�ֵ
rb=(bayer(ii-1,jj)+bayer(ii+1,jj))/2+ ...
    (2*bayer(ii,jj)-rgb(ii-1,jj,2)-rgb(ii+1,jj,2))/2;
end

function rb=RBH(ii,jj,bayer,rgb)    %��/��ɫ����ʹ����������R/B-G�ռ�����Բ�ֵ
rb=(bayer(ii,jj-1)+bayer(ii,jj+1))/2+ ...
    (2*bayer(ii,jj)-rgb(ii,jj-1,2)-rgb(ii,jj+1,2))/2;
end

% ��ɫ(��ɫ)�����㴦����ɫ(��ɫ)�������ؽ�
function grad=D_45(ii,jj,bayer,rgb)   %���������ݶȼ���
grad=abs(bayer(ii-1,jj+1)-bayer(ii+1,jj-1))+ ...
    abs(2*rgb(ii,jj,2)-rgb(ii-1,jj+1,2)-rgb(ii+1,jj-1,2));
end

function grad=D_135(ii,jj,bayer,rgb)  %���������ݶȼ���
grad=abs(bayer(ii-1,jj-1)-bayer(ii+1,jj+1))+ ...
    abs(2*rgb(ii,jj,2)-rgb(ii-1,jj-1,2)-rgb(ii+1,jj+1,2));
end

function rb=RB45(ii,jj,bayer,rgb)     %D_45<D_135
rb=(bayer(ii-1,jj+1)+bayer(ii+1,jj-1))/2+ ...
    (2*rgb(ii,jj,2)-rgb(ii-1,jj+1,2)-rgb(ii+1,jj-1,2))/2;
end

function rb=RB135(ii,jj,bayer,rgb)    %D_45>D_135
rb=(bayer(ii-1,jj-1)+bayer(ii+1,jj+1))/2+ ...
    (2*rgb(ii,jj,2)-rgb(ii-1,jj-1,2)-rgb(ii+1,jj+1,2))/2;
end

function rb=RBM(ii,jj,bayer,rgb)      %D_45=D_135
rb=(RB45(ii,jj,bayer,rgb)+RB135(ii,jj,bayer,rgb))/2;
end