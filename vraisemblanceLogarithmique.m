function val = vraisemblanceLogarithmique(X,mu,sigma,pi)
% vraisemblanceLogarithmique calcule la vraisemblance logarithmique du jeu 
%  de donnees connaissant les parametres du modele
% ARGUMENTS:
%   X: données sur lesquels calculer la vraisemblance logarithmique
%   mu: matrice de dimension Kxdimension ayant pour ligne les moyennes de chaque distribution
%   sigma: matrices de variance-covariance de chaque distribution, les unes
%       à la suite des autres
%   pi: matrice colonne de probabilité a priori pour chaque ditribution
% SORTIE:
%   val: vraisemblance logarithmique du jeu de donnees (Scalaire)

	% Paramètres
	s = 0; % Variable temporaire pour stocker la vraissemblance logarithmique temporaire
	n = size(X, 1); % Nombre de points dans la base de donnees
	K = size(mu,1); % Nombre de cluster
    dimension = size(mu,2);
    
	% Iteration pour obtenir la somme des termes de la vraisemblance logarithmique
	for i=1:n
		tempLog = 0;
		for j=1:K
			tempLog = tempLog + pi(j) * mvnpdf(X(i,:), mu(j,:), sigma(:,dimension*(j-1)+1:dimension*j));
		end
		s = s + log(tempLog);
	end
	val = s;
end
