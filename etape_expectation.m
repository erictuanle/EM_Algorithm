function T = etape_expectation(X,pi,mu,sigma)
% Expectation permet de faire l'�tape d'Expectation de l'algorithme EM
% ARGUMENTS:
%   X: donn�es sur lesquels faire tourner l'algorithme EM
%   mu: matrice de dimension Kxdimension ayant pour ligne les moyennes de chaque distribution
%   sigma: matrices de variance-covariance de chaque distribution, les unes
%       � la suite des autres
%   pi: matrice colonne de probabilit� a priori pour chaque ditribution
% SORTIE:
%   T: matrice de dimension dimension*K de probabilit� que yi viennent de la
%       distribution k sachant xi

% Param�tres
K = size(mu,1);
n = size(X,1);
T = zeros(n,K);
dimension = size(X,2);

% D�termination de l'Expectation
for i=1:n
    for j=1:K
        denominateur=0;
        for k=1:K
            denominateur = denominateur + pi(k) * mvnpdf(X(i,:),mu(k,:),sigma(:,dimension*(k-1)+1:dimension*k));
        end
        T(i, j) = pi(j) * mvnpdf(X(i,:),mu(j,:),sigma(:,dimension*(j-1)+1:dimension*j))/denominateur;
    end
end