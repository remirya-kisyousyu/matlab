clear %変数全消去
close all %開いているfigureを全て閉じる
clc %コマンドウィンドウに書かれているテキストを全部消去

% X = [ 1 2 3 4 ; 5 6 7 8 ; 9 10 11 12 ; 13 14 15 16 ];
X = im2double( imread( 'Peppers512rgb.png' ) );
X = rgb2gray( X );
lm = 1;

%fai : 画像を欠損させる行列[512*512, 512*512]
fai = eye(size(X, 1), size(X, 1));
fai = fai .* rand(size(fai));
fai = fai > 0.2;
fai = sparse(kron(speye(size(X, 1), size(X, 2)), fai));

x = X(:);
%nX : 欠損画像
nX = fai * x;
%reshape(X, 512, 512);
% figure(1)
% imshow(X)

% nX = reshape(nX, 512, 512);
% figure(2)
% imshow(nX)

%1/2*(x - nX)^2 + lm*(Dh*X)^2 + lm*(Dv*X)^2
%(x - nX) + 2*lm*Dh.'(Dh*X) + 2*lm*Dv.'(Dv*X)
%% ③線形代数を用いた差分フィルタ処理（線形代数を用いる->画像をベクトルデータ（1次元配列）として取り扱う考え方）

% 2次元配列の1次元ベクトルデータの方法：2つ
% ①コロン演算子を使用 x = X(:);
% ②関数reshapeを使用 x = reshape( 入力データX , 変形したいサイズ[縦次元,横次元] );

% 1次元データから2次元配列に戻す方法：1つ　
% ①関数reshapeを使用 X = reshape( 入力データX , 変形したいサイズ[縦次元,横次元] );
% 各R, G, B成分をベクトル化
%x = reshape( X , [ size(X,1)*size(X,2) , 1  ] );

% 行列作成 Dv, Dh
I = speye(size(X,1),size(X,1));

Dv0 = -I + circshift(I,1,2);

Dv = sparse(kron(I,Dv0));
Dh = sparse(kron(Dv0,I));

%i = speye(size(nx,1),size(nx,1));

%推定画像
nx = nX(:);
xx = zeros(size(nx, 1), 1);
alpha = 0.1;

%最小二乗法による欠損補間画像の推定
denoiseX = (fai + 2*lm*(Dh.')*Dh + 2*lm*(Dv.')*Dv)\nx;

restoreImage = reshape(denoiseX,size(X,1),size(X,2));
figure(3)
imshow(restoreImage)

%argmin 1/2||fai*x - y||^2 + lamda * ||Dv * x||^2 + lamda * ||Dh * x||^2
%1/2||fai*x - y||^2 -> (fai.') * (fai * x - y)
%lamda * ||Dv * x||^2 -> 2 * (Dv.') * (Dv * x)
%lamda * ||Dh * x||^2 -> 2 * (Dh.') * (Dh * x)
%x = X(:);

%最急降下法の反復を用いた最小二乗法による欠損補間画像の推定
for i = 0 : 100
    %d_dx : 微分係数
    d_dx = (fai.') * (fai * xx - nx) + 2 * (Dv.') * (Dv * xx) + 2 * (Dh.') * (Dh * xx);
    xx = xx - alpha * d_dx; %xの更新
end

% denX2 = (1/2) * (x - nx).^2 + (Dv * x).^2 + (Dh * x).^2;
% denX2re = reshape(denX2, size(X, 1), size(X, 2));
restoreImage = reshape(xx, size(X, 1), size(X, 2));
figure(4)
imshow(restoreImage)

%norm(xx - denXre)