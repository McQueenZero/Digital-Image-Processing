%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ��ȡͼ����״����
%%-------------------------------------------------------------------------
%% ��ȡͼ����״����
clc, clear, close all
filename = 'building.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);

% Hu�أ�����أ�������������˹.MATLAB������ͼ����,P.353��
phi = invmoments(imSrc);
phi = abs(log(phi));
disp('����ͼȡ��������ֵ��Hu�����£�')
disp('    phi_1    phi_2     phi_3     phi_4     phi_5     phi_6     phi_7')
disp(phi)

% ROI������������˹.MATLAB������ͼ����,P.114��
[B, c, r] = roipoly(imSrc); %����ʽѡ�����Ȥ��ROI������
imDst = uint8(double(imSrc).*double(B));
figure('Name','����Ȥ����','NumberTitle','off')
imshow(imDst)

[p, npix] = histroi(imSrc, c, r);
figure('Name','��ģB�����ֱ��ͼ','NumberTitle','off')
bar(p,1) %��ģB�����ֱ��ͼ

phi = invmoments(imDst);
phi = abs(log(phi));
disp('����Ȥ����ȡ��������ֵ��Hu�����£�')
disp('    phi_1    phi_2     phi_3     phi_4     phi_5     phi_6     phi_7')
disp(phi)

%% ��ȡͼ����״������Ƕ����
function phi = invmoments(F)
% INVMOMENTS Compute invariant moments of image.
%   PHI = INVMOMENTS(F) computes the moment invariants of the image
%   F. PHI is a seven-element row vector containing the moment
%   invariants as defined in equations (11.3-11.7) through (11.3-23) of
%   Gonzalez and Woods, Digital Image Processing, 2nd Ed.
% 
%   F must be a 2-D, real, nonsparse, numeric or logical matrix.
if ndims(F)~=2 || issparse(F) || ~isreal(F) || ~(isnumeric(F) || islogical(F))
    error('F must be a 2-D, real, nonsparse, numeric or logical matrix.');
end
F = double(F);
phi = compute_phi(compute_eta(compute_m(F)));
end
%-------------------------------------------------------------------------%
function m = compute_m(F)
[M, N] = size(F);
[x, y] = meshgrid(1:N, 1:M);
% Turn x, y, and F into column vectors to make the summations a bit
% easier to compute in the following.
x = x(:);
y = y(:);
F = F(:);
% DIP equation (11.3-12)
m.m00 = sum(F);
% Protect against divide-by-zero warnings.
if m.m00 == 0
    m.m00 = eps;
end
% The other central moments:
m.m10 = sum(x.*F);
m.m01 = sum(y.*F);
m.m11 = sum(x.*y.*F);
m.m20 = sum(x.^2.*F);
m.m02 = sum(y.^2.*F);
m.m30 = sum(x.^3.*F);
m.m03 = sum(y.^3.*F);
m.m12 = sum(x.*y.^2.*F);
m.m21 = sum(x.^2.*y.*F);
end
%-------------------------------------------------------------------------%
function e = compute_eta(m)
% DIP equations (11.3-14) through (11.3-16).
xbar = m.m10/m.m00;
ybar = m.m01/m.m00;
e.eta11 = (m.m11-ybar*m.m10)/m.m00^2;
e.eta20 = (m.m20-xbar*m.m10)/m.m00^2;
e.eta02 = (m.m02-ybar*m.m01)/m.m00^2;
e.eta30 = (m.m30-3*xbar*m.m20+2*xbar^2*m.m10)/m.m00^2.5;
e.eta03 = (m.m03-3*ybar*m.m02+2*ybar^2*m.m01)/m.m00^2.5;
e.eta21 = (m.m21-2*xbar*m.m11-ybar*m.m20+2*xbar^2*m.m01)/m.m00^2.5;
e.eta12 = (m.m12-2*ybar*m.m11-xbar*m.m02+2*ybar^2*m.m10)/m.m00^2.5;
end
%-------------------------------------------------------------------------%
function phi = compute_phi(e)
% DIP equations (11.3-17) through (11.3-23)
phi(1) = e.eta20 + e.eta02;
phi(2) = (e.eta20 - e.eta02)^2 + 4*e.eta11^2;
phi(3) = (e.eta30 - 3*e.eta12)^2 + (3*e.eta21 - e.eta03)^2;
phi(4) = (e.eta30 + e.eta12)^2 + (e.eta21 + e.eta03)^2;
phi(5) = (e.eta30 - 3*e.eta12)*(e.eta30 + e.eta12)*((e.eta30 + e.eta12)^2 - 3*(e.eta21 + e.eta03)^2) + ...
         (3*e.eta21 - e.eta03)*(e.eta21 + e.eta03)*(3*(e.eta30 + e.eta12)^2 - (e.eta21 + e.eta03)^2);
phi(6) = (e.eta20 - e.eta02)*((e.eta30 + e.eta12)^2 - (e.eta21 + e.eta03)^2) + ...
         4*e.eta11*(e.eta30 + e.eta12)*(e.eta21 + e.eta03);
phi(7) = (3*e.eta21 - e.eta03)*(e.eta30 + e.eta12)*((e.eta30 + e.eta12)^2 - 3*(e.eta21 + e.eta03)^2) + ...
         (3*e.eta12 - e.eta30)*(e.eta21 + e.eta03)*(3*(e.eta30 + e.eta12)^2 - (e.eta21 + e.eta03)^2);
end

function [p, npix] = histroi(f, c, r)
% HISTROI Computes the histogram of an ROI in an image.
%   [P, NPIX] = HISTROI(F, C, R) computes the histogram, P, of a 
%   polygonal region of interesst (ROI) in image F. The polygonal
%   region is defined by the column and row coordinates of its
%   vertices, which are speciried (sequentially) in vectors C and R,
%   respectively. All pixels of F must be >= 0. Parameter NPIX is the 
%   number of pixxels in the polygonal region.
% Generate the binary mask image.
B = roipoly(f, c, r);
% Compute the histogram of the pixels in the ROI.
p = imhist(f(B));
% Obtain the number of pixels in the ROI if requested in the output.
if nargout > 1
    npix = sum(B(:));
end
end

