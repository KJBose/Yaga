(function() {
'use strict';

angular.module('public')
.config(routeConfig);

/**
 * Configures the routes and views
 */
routeConfig.$inject = ['$stateProvider'];
function routeConfig ($stateProvider) {
  // Routes
  $stateProvider
    .state('public', {
      absract: true,
      templateUrl: 'src/public/public.html'
    })
    .state('public.home', {
      url: '/',
      templateUrl: 'src/public/home/home.html'
    })
    .state('public.menu', {
      url: '/menu',
      templateUrl: 'src/public/menu/menu.html',
      controller: 'MenuController',
      controllerAs: 'menuCtrl',
      resolve: {
        menuCategories: ['MenuService', function(MenuService) {
          return MenuService.getCategories();
        }]
      }
    })
    .state('public.signup', {
      url: '/signup',
      templateUrl: 'src/public/sign-up/signup.html',
      controller: 'SignUpFormController',
      controllerAs: 'signer'
    })
    .state('public.signin', {
      url: '/signin',
      templateUrl: 'src/public/sign-up/signin.html',
      controller: 'SignInFormController',
      controllerAs: 'signerIn'
    })
    /*.state('public.favMenu', {
      url: '/favMenu',
      templateUrl: 'src/public/fav-menu/fav-menu.html',
      controller: 'FavMenuController',
      controllerAs: 'FavMenupCtrl',
      resolve: {
        fovourite: ['MenuService', function(MenuService) {
          return MenuService.getMenuItems(category);
        }]
      }
    })*/
    .state('public.myinfo', {
      url: '/myinfo',
      templateUrl: 'src/public/my-info/myinfo.html',
      controller: 'MyInfoController',
      controllerAs: 'myinfoCtrl',
      resolve: {
        userInfo: ['UserDataService', function(UserDataService) {
          return UserDataService.getUserData();
        }]
      }
    })
    .state('public.menuitems', {
      url: '/menu/{category}',
      templateUrl: 'src/public/menu-items/menu-items.html',
      controller: 'MenuItemsController',
      controllerAs: 'menuItemsCtrl',
      resolve: {
        menuItems: ['$stateParams','MenuService', function ($stateParams, MenuService) {
          return MenuService.getMenuItems($stateParams.category);
        }]
      }
    });
}
})();
