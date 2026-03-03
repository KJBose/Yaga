(function () {
"use strict";

angular.module('common')
.controller('AuthNavController', AuthNavController);

AuthNavController.$inject = ['$state', '$rootScope', 'AuthService'];
function AuthNavController($state, $rootScope, AuthService) {
  var nav = this;
  var currentStateName = '';

  nav.isAuthenticated = function () {
    return AuthService.isAuthenticated();
  };

  nav.isState = function (stateName) {
    return currentStateName === stateName;
  };

  nav.isProfileActive = function () {
    return currentStateName === 'public.profile' || currentStateName === 'public.userProfile';
  };

  nav.signOut = function () {
    AuthService.signOut()
      .finally(function () {
        $state.go('public.index');
      });
  };

  $rootScope.$on('$stateChangeSuccess', function (_event, toState) {
    currentStateName = (toState && toState.name) || '';
  });

  currentStateName = ($state.current && $state.current.name) || '';
}

})();
