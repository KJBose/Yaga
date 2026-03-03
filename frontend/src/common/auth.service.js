(function () {
"use strict";

angular.module('common')
.service('AuthService', AuthService);


AuthService.$inject = ['$http', '$window', '$q', 'AuthApiPath'];
function AuthService($http, $window, $q, AuthApiPath) {
  var authService = this;
  var TOKEN_KEY = 'yaga_auth_token';
  var cachedUser = null;

  authService.signUp = function (payload) {
    return $http.post(AuthApiPath + '/auth/signup', payload).then(function (response) {
      saveSession(response.data);
      return response.data;
    });
  };

  authService.signIn = function (payload) {
    return $http.post(AuthApiPath + '/auth/signin', payload).then(function (response) {
      saveSession(response.data);
      return response.data;
    });
  };

  authService.signOut = function () {
    return $http.delete(AuthApiPath + '/auth/signout')
      .finally(function () {
        clearSession();
      });
  };

  authService.isAuthenticated = function () {
    return !!$window.localStorage.getItem(TOKEN_KEY);
  };

  authService.getCurrentUser = function () {
    if (!authService.isAuthenticated()) {
      return $q.reject({ error: 'Not authenticated' });
    }

    if (cachedUser) {
      return $q.when(cachedUser);
    }

    return $http.get(AuthApiPath + '/auth/me').then(function (response) {
      cachedUser = response.data;
      return cachedUser;
    });
  };

  function saveSession(data) {
    $window.localStorage.setItem(TOKEN_KEY, data.token);
    cachedUser = {
      user: data.user,
      profile: data.profile,
      help_counts: data.help_counts,
      appreciation: data.appreciation
    };
  }

  function clearSession() {
    $window.localStorage.removeItem(TOKEN_KEY);
    cachedUser = null;
  }
}

})();
