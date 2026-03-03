(function () {
"use strict";

angular.module('public')
.controller('SignInFormController', SignInFormController);

SignInFormController.$inject = ['$state', 'AuthService'];
function SignInFormController($state, AuthService) {
  var $ctrl = this;
  $ctrl.user = {
    email: '',
    password: ''
  };
  $ctrl.error = '';
  $ctrl.processing = false;

  $ctrl.submit = function () {
    $ctrl.error = '';
    $ctrl.processing = true;

    AuthService.signIn($ctrl.user)
      .then(function () {
        return $state.go('public.profile');
      })
      .catch(function (error) {
        $ctrl.error = (error.data && error.data.error) || 'Unable to login.';
      })
      .finally(function () {
        $ctrl.processing = false;
      });
  };
}

})();
