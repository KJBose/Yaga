(function () {
"use strict";

angular.module('public')
.controller('SignUpFormController', SignUpFormController);

SignUpFormController.$inject = ['$state', 'AuthService'];
function SignUpFormController($state, AuthService) {
  var $ctrl = this;

  $ctrl.userTypes = ['Individual', 'NPO', 'GOV'];
  $ctrl.user = {
    first_name: '',
    last_name: '',
    email: '',
    password: '',
    password_confirmation: '',
    phone: '',
    user_type: 'Individual',
    organization_name: '',
    bio: ''
  };
  $ctrl.error = '';
  $ctrl.success = '';
  $ctrl.processing = false;

  $ctrl.submit = function () {
    $ctrl.error = '';
    $ctrl.success = '';
    $ctrl.processing = true;

    AuthService.signUp($ctrl.user)
      .then(function () {
        $ctrl.success = 'Account created. Redirecting to your profile...';
        return $state.go('public.profile');
      })
      .catch(function (error) {
        var apiErrors = error && error.data && error.data.errors;
        $ctrl.error = apiErrors ? apiErrors.join(', ') : 'Unable to create account.';
      })
      .finally(function () {
        $ctrl.processing = false;
      });
  };
}

})();
