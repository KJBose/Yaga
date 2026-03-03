(function () {
"use strict";

angular.module('common')
.service('UserDataService', UserDataService);


UserDataService.$inject = ['$http', '$q', 'AuthApiPath'];
function UserDataService($http, $q, AuthApiPath) {
  var userService = this;
  userService.cachedDashboard = null;

  userService.getProfile = function () {
    return $http.get(AuthApiPath + '/profile').then(function (response) {
      return response.data;
    });
  };

  userService.updateProfile = function (profile) {
    return $http.patch(AuthApiPath + '/profile', { profile: profile }).then(function (response) {
      return response.data;
    });
  };

  userService.getPublicProfile = function (profileId) {
    return $http.get(AuthApiPath + '/profiles/' + profileId).then(function (response) {
      return response.data;
    });
  };

  userService.getTransformations = function () {
    return $http.get(AuthApiPath + '/transformations').then(function (response) {
      return response.data;
    });
  };

  userService.getAppreciations = function () {
    return $http.get(AuthApiPath + '/appreciations').then(function (response) {
      return response.data;
    });
  };

  userService.getDashboard = function () {
    return $q
      .all([
        userService.getProfile(),
        userService.getTransformations().catch(function () {
          return { totals: { completed: 0, led: 0, contributed: 0 }, transformations: [] };
        }),
        userService.getAppreciations().catch(function () {
          return { appreciations: [] };
        })
      ])
      .then(function (results) {
        var profileData = results[0];
        var transformationsData = results[1];
        var appreciationsData = results[2];

        userService.cachedDashboard = {
          profile: profileData.profile,
          help_counts: profileData.help_counts,
          appreciation: profileData.appreciation,
          transformations: transformationsData.transformations,
          transformation_totals: transformationsData.totals,
          appreciations: appreciationsData.appreciations
        };

        return userService.cachedDashboard;
      });
  };

  userService.getCachedDashboard = function () {
    return userService.cachedDashboard;
  };
}

})();
