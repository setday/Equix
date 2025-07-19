"""Dependency injection container for the Equix application."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, TypeVar, get_type_hints

from src.core.exceptions import ConfigurationError

T = TypeVar("T")


class Container:
    """Simple dependency injection container."""

    def __init__(self) -> None:
        """Initialize the container."""
        self._services: dict[type, Any] = {}
        self._factories: dict[type, Callable[..., Any]] = {}
        self._singletons: dict[type, Any] = {}
        self._initializing: set[type] = set()

    def register(self, interface: type[T], implementation: type[T] | T) -> None:
        """Register a service implementation.
        
        Args:
            interface: The interface type
            implementation: The implementation class or instance
        """
        if isinstance(implementation, type):
            # Register as a factory
            self._factories[interface] = implementation
        else:
            # Register as singleton instance
            self._singletons[interface] = implementation

    def register_singleton(self, interface: type[T], implementation: type[T]) -> None:
        """Register a service as singleton.
        
        Args:
            interface: The interface type
            implementation: The implementation class
        """
        self._factories[interface] = implementation

    def register_instance(self, interface: type[T], instance: T) -> None:
        """Register a service instance.
        
        Args:
            interface: The interface type
            instance: The service instance
        """
        self._singletons[interface] = instance

    def get(self, interface: type[T]) -> T:
        """Get a service instance.
        
        Args:
            interface: The interface type
            
        Returns:
            Service instance
            
        Raises:
            ConfigurationError: If service is not registered
        """
        # Check if already instantiated as singleton
        if interface in self._singletons:
            return self._singletons[interface]

        # Check for circular dependencies
        if interface in self._initializing:
            raise ConfigurationError(
                f"Circular dependency detected for {interface}",
                "CIRCULAR_DEPENDENCY",
            )

        # Check if factory is registered
        if interface not in self._factories:
            raise ConfigurationError(
                f"Service {interface} is not registered",
                "SERVICE_NOT_REGISTERED",
            )

        # Create instance using factory
        self._initializing.add(interface)
        try:
            factory = self._factories[interface]
            instance = self._create_instance(factory)
            
            # Store as singleton for future requests
            self._singletons[interface] = instance
            return instance
        finally:
            self._initializing.discard(interface)

    def _create_instance(self, factory: Callable[..., T]) -> T:
        """Create an instance using dependency injection.
        
        Args:
            factory: The factory class or callable
            
        Returns:
            Created instance
        """
        # Handle callable factories vs class factories differently
        if not inspect.isclass(factory):
            # It's a callable factory
            return factory()
        
        print("3")
            
        # Get constructor type hints
        type_hints = get_type_hints(factory.__init__)
        
        # Remove 'self' and 'return' from hints
        dependencies = {
            name: hint for name, hint in type_hints.items() 
            if name not in ('self', 'return')
        }

        # Resolve dependencies
        kwargs = {}
        # for name, dep_type in dependencies.items():
        #     kwargs[name] = self.get(dep_type)

        return factory(**kwargs)

    async def get_async(self, interface: type[T]) -> T:
        """Get a service instance asynchronously.
        
        Args:
            interface: The interface type
            
        Returns:
            Service instance
        """
        # Run in thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, interface)

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._initializing.clear()

    def is_registered(self, interface: type) -> bool:
        """Check if a service is registered.
        
        Args:
            interface: The interface type
            
        Returns:
            True if registered, False otherwise
        """
        return interface in self._factories or interface in self._singletons


# Global container instance
container = Container()
