function [X, classes] = gaussianmixture(n,pi,mu,sigma)
%gaussianmixture simule un jeu de donn�es issu d'un mod�le de m�lange
%   gaussien de K composantes en dimension 2
%   INPUTS:
%       n: nombre d'�chantillons
%       pi: probabilit� a priori des classes
%       mu: moyenne des classes
%       sigma: matrice de covariance des classes
%   OUTPUTS:
%       X: jeu de donn�es cherch�
%       classes: association entre les donn�es et leur classe

% Pour connaitre la dimension des �chantillons cherch�s
dimension = size(mu,2);
% Initialisation
X = zeros(n,dimension);
classes = zeros(n,1);

% Pour tous les �chantillons
for i=1:n
    % On choisit au hasard la classe de l'�chantillon
    r = rand;
    k = sum(r >= cumsum([0, pi']));
    classes(i) = k;
    % On tire un �chantillon al�atoire de la classe choisie au hasard
    echantillon = mvnrnd(mu(k,:),sigma(:,dimension*(k-1)+1:dimension*k));
    % On ajoute l'�chantillon au jeu de donn�es
    X(i,:) = echantillon;
end

end

