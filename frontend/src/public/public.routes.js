(function() {
'use strict';

angular.module('public')
.config(routeConfig);


routeConfig.$inject = ['$stateProvider'];
function routeConfig($stateProvider) {
  $stateProvider
    .state('public', {
      abstract: true,
      templateUrl: 'src/public/public.html'
    })
    .state('public.index', {
      url: '/',
      templateUrl: 'src/public/index/index.html?v=20260303m'
    })
    .state('public.home', {
      url: '/home',
      templateUrl: 'src/public/home/home.html?v=20260303k',
      controller: 'HomeController',
      controllerAs: 'homeCtrl',
      onEnter: ['AuthService', '$state', function(AuthService, $state) {
        if (!AuthService.isAuthenticated()) {
          $state.go('public.index');
        }
      }]
    })
    .state('public.menu', {
      url: '/support',
      templateUrl: 'src/public/menu/menu.html?v=20260313w',
      controller: 'MenuController',
      controllerAs: 'menuCtrl',
      resolve: {
        helps: ['HelpService', function(HelpService) {
          return HelpService.getPrioritizedHelps().catch(function () {
            return [];
          });
        }]
      }
    })
    .state('public.chat', {
      url: '/chat?withProfileId&chatId',
      templateUrl: 'src/public/chat/chat.html?v=20260312',
      controller: 'ChatController',
      controllerAs: 'chatCtrl'
    })
    .state('public.createAccount', {
      url: '/create-account',
      templateUrl: 'src/public/sign-up/signup.html?v=20260312',
      controller: 'SignUpFormController',
      controllerAs: 'signer'
    })
    .state('public.signup', {
      url: '/signup',
      onEnter: ['$state', function($state) {
        $state.go('public.createAccount');
      }]
    })
    .state('public.signin', {
      url: '/signin',
      templateUrl: 'src/public/sign-in/signin.html?v=20260312',
      controller: 'SignInFormController',
      controllerAs: 'signerIn'
    })
    .state('public.profile', {
      url: '/profile',
      templateUrl: 'src/public/my-info/myinfo.html?v=20260303d',
      controller: 'MyInfoController',
      controllerAs: 'myinfoCtrl',
      resolve: {
        userInfo: ['AuthService', 'UserDataService', '$q', function(AuthService, UserDataService, $q) {
          if (!AuthService.isAuthenticated()) {
            return $q.when(null);
          }

          return UserDataService.getDashboard()
            .catch(function (error) {
              if (error && error.status === 401) {
                return AuthService.signOut()
                  .then(function () {
                    return null;
                  })
                  .catch(function () {
                    return null;
                  });
              }

              return null;
            });
        }]
      }
    })
    .state('public.userProfile', {
      url: '/users/:profileId',
      templateUrl: 'src/public/my-info/myinfo.html?v=20260303d',
      controller: 'MyInfoController',
      controllerAs: 'myinfoCtrl',
      resolve: {
        userInfo: ['$stateParams', 'UserDataService', function($stateParams, UserDataService) {
          return UserDataService.getPublicProfile($stateParams.profileId).catch(function (error) {
            if (error && error.status === 403) {
              return {
                restricted: true,
                restricted_message: (error.data && error.data.error) || 'This profile is private based on user settings.'
              };
            }
            return null;
          });
        }]
      }
    })
    .state('public.myinfo', {
      url: '/myinfo',
      onEnter: ['$state', function($state) {
        $state.go('public.profile');
      }]
    });
}
})();
