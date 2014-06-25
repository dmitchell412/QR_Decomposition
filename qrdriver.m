%function driver

clear all
close all
format shortg

tol = 0.0001;
upbound = 1000;
nDim_image = 2;
nDim_matrix = 4;

h_root = complex(zeros(nDim_matrix-1,nDim_image,nDim_image),zeros(nDim_matrix-1,nDim_image,nDim_image));
h_poly = complex(randn(nDim_matrix,nDim_image,nDim_image),randn(nDim_matrix,nDim_image,nDim_image));
%h_a = randn(nDim_matrix,nDim_matrix,nDim_image,nDim_image);
%h_Q = zeros(nDim_matrix,nDim_matrix,nDim_image,nDim_image);
%h_R = zeros(nDim_matrix,nDim_matrix,nDim_image,nDim_image);

%for i = 1:nDim_image
%    for j = 1:nDim_image
%        h_a(:,:,i,j) = [1,1,0;1,0,1;0,1,1];
%    end
%end

%for i = 1:nDim_image
%    for j = 1:nDim_image
%        h_poly(:,i,j) = [1,-6,-72,-27];
%    end
%end

% transfer data to device
d_poly  = gpuArray( h_poly );
d_root  = gpuArray( h_root );
%d_a  = gpuArray( h_a );
%d_Q  = gpuArray( h_Q );
%d_R  = gpuArray( h_R );

qrdptx = parallel.gpu.CUDAKernel('qrd.ptx', 'qrd.cu');
threadsPerBlock = 256;
npixel = nDim_image*nDim_image;
qrdptx.ThreadBlockSize=[threadsPerBlock, 1, 1];
%blocksPerGrid = (npixel + threadsPerBlock -1) / threadsPerBlock;
%blocksPerGrid = (npixel  * threadsPerBlock - 1) / threadsPerBlock;
blocksPerGrid = ceil(npixel/threadsPerBlock);
qrdptx.GridSize=[blocksPerGrid, 1, 1];
%qrdptx.GridSize=[ceil(blocksPerGrid), 1, 1];
%qrdptx.GridSize=[256, 1, 1];

[dpolyrealout,dpolyimagout,drootrealout,drootimagout] = feval(qrdptx,real(d_poly),imag(d_poly),real(d_root),imag(d_root),tol,upbound,nDim_image,nDim_matrix);
%[daout,dQout,dRout] = feval(qrdptx,d_poly,d_Q,d_R,nDim_image,nDim_matrix);

%for i=1:nDim_image
%    for j=1:nDim_image
%        xtest(:,i,j)=h_A(:,:,i,j)\h_b(:,i,j);
%        xcheck = norm(xtest(:,i,j)-dxout(:,i,j));
%        if xcheck >= 1.0e-8 
%            xcheck
%        end
%    end
%end

%exit
