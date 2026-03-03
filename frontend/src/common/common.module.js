(function() {
"use strict";

angular.module('common', [])
.constant('ApiPath', 'http://localhost:3000')
.constant('AuthApiPath', 'http://localhost:3000/api/v1')
.constant('ChainConfig', {
  defaultChainId: 11155111,
  contractAddress: '',
  useChainActionsByDefault: false
})
.config(config);

config.$inject = ['$httpProvider'];
function config($httpProvider) {
  $httpProvider.interceptors.push('loadingHttpInterceptor');
}

})();
