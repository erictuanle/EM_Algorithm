function [mu,sigma,pi] = etape_maximisation(X,T)
% Maximisation permet de faire l'étape de Maximisation de l'algorithme EM
% ARGUMENTS:
%   X: données sur lesquels faire tourner l'algorithme EM
%   T: matrice de dimension dimension*K de probabilité que yi viennent de la
%       distribution k sachant xi
% SORTIE:
%   mu: matrice de dimension Kxdimension ayant pour ligne les moyennes de chaque distribution
%   sigma: matrices de variance-covariance de chaque distribution, les unes
%       à la suite des autres
%   pi: matrice colonne de probabilité a priori pour chaque ditribution
 
% Paramètre
n = size(T,1);
K = size(T,2); 
dimension = size(X,2);
mu = zeros(K,dimension);

% Calcul des proportions de chaque distribution
pi = zeros(K,1);
nk = sum(T,1);
for i=1:K
    pi(i) = nk(i)/n;
end

% Calcul de l'esperance mu
for i=1:K
    somme = 0;
    for j=1:n
        somme = somme + T(j,i)*X(j, :);
    end
    somme = somme/nk(i);
    mu(i,:) = somme(:);
end

% Calcul de l'ecart-type sigma
sigma = zeros(dimension,dimension*K);
for i=1:K
    somme = 0;
    for j=1:n
        ecartMoyenne = X(j,:) - mu(i,:);
        somme = somme + T(j,i) * (ecartMoyenne' * ecartMoyenne);
    end
    somme = somme/nk(i);
    sigma(:,dimension*(i-1)+1:dimension*i) = somme;
end
